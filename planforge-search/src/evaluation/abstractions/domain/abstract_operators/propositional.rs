//! The propositional cross product of a concrete operator: one skeleton for
//! every abstract value its effects without a precondition may change from.

use planforge_sas::numeric_task::{
    AbstractNumericTask, ExplicitFact, ExplicitValueIndex, Operator, VariableIndex,
};

use super::transition_vars_match;
use crate::evaluation::abstractions::domain::factory::DomainAbstractionFactory;
use crate::evaluation::abstractions::domain::layout::StateLayout;

/// The propositional part of an abstract operator: the prevail conditions,
/// and the preconditions and effects of the variables it changes. Each list
/// is strictly ascending by variable, and `pre_pairs` and `eff_pairs` name
/// the same variables.
#[derive(Debug, Clone)]
pub(super) struct PropositionalSkeleton {
    pub(super) prev_pairs: Vec<ExplicitFact>,
    pub(super) pre_pairs: Vec<ExplicitFact>,
    pub(super) eff_pairs: Vec<ExplicitFact>,
}

/// Builds the propositional skeletons of concrete operators, with
/// per-variable buffers reused across operators.
pub(super) struct PropositionalSkeletons<'a> {
    factory: &'a DomainAbstractionFactory,
    /// The abstract value the current operator's preconditions give each
    /// variable; all `None` between operators.
    precondition_on_var: Vec<Option<ExplicitValueIndex>>,
    /// Whether the current operator changes each variable's abstract value;
    /// all `false` between operators.
    changed_var: Vec<bool>,
}

impl<'a> PropositionalSkeletons<'a> {
    pub(super) fn new(
        factory: &'a DomainAbstractionFactory,
        task: &dyn AbstractNumericTask,
    ) -> Self {
        let num_variables = task.get_num_variables();
        Self {
            factory,
            precondition_on_var: vec![None; num_variables],
            changed_var: vec![false; num_variables],
        }
    }

    fn variable_is_trivial(&self, var_id: VariableIndex) -> bool {
        self.factory.layout().domain_sizes()[var_id.index()] <= 1
    }

    fn abstract_value(
        &self,
        var_id: VariableIndex,
        concrete_value: ExplicitValueIndex,
    ) -> ExplicitValueIndex {
        self.factory.domain_mapping()[var_id.index()][concrete_value.index()]
    }

    /// The skeletons of `op`, in cross-product order. An operator that
    /// changes no abstract propositional value and has no assignment effect
    /// has none.
    pub(super) fn of_operator(
        &mut self,
        task: &dyn AbstractNumericTask,
        op: &Operator,
    ) -> Vec<PropositionalSkeleton> {
        let preconditions = op.preconditions();
        for pre in preconditions {
            let var_id = pre.var_index();
            self.precondition_on_var[var_id.index()] = Some(if self.variable_is_trivial(var_id) {
                ExplicitValueIndex::new(0)
            } else {
                self.abstract_value(var_id, pre.value_index())
            });
        }

        let mut facts = PropositionalFactScratch::default();
        let mut effects_without_pre: Vec<ExplicitFact> = Vec::new();
        for eff in op.effects() {
            let var_id = eff.var_id();
            if self.variable_is_trivial(var_id) {
                continue;
            }
            debug_assert!(!task.numeric_conditions().is_condition_var(var_id));
            let abs_val = self.abstract_value(var_id, eff.value());
            let fact = ExplicitFact::propositional(var_id.index(), abs_val.index());
            match self.precondition_on_var[var_id.index()] {
                Some(pre_val) if pre_val != abs_val => {
                    self.changed_var[var_id.index()] = true;
                    facts.eff_pairs.push(fact);
                }
                Some(_) => {}
                None => effects_without_pre.push(fact),
            }
        }

        // Condition variables are never refined, so they are trivial here: a
        // comparison precondition only filters the numeric partition
        // combinations the operator applies in.
        for pre in preconditions {
            let var_id = pre.var_index();
            if self.variable_is_trivial(var_id) {
                continue;
            }
            let abs_val = self.abstract_value(var_id, pre.value_index());
            if self.changed_var[var_id.index()] {
                facts.pre_pairs.push(ExplicitFact::in_namespace(
                    pre.namespace(),
                    var_id.index(),
                    abs_val.index(),
                ));
            } else {
                facts
                    .prev_pairs
                    .push(ExplicitFact::propositional(var_id.index(), abs_val.index()));
            }
        }

        for pre in preconditions {
            self.precondition_on_var[pre.var()] = None;
        }
        for eff in &facts.eff_pairs {
            self.changed_var[eff.var()] = false;
        }

        let mut skeletons = Vec::new();
        PropositionalCrossProduct {
            effects_without_pre: &effects_without_pre,
            has_assignment_effects: !op.assignment_effects().is_empty(),
            layout: self.factory.layout(),
        }
        .multiply_out(0, &mut facts, &mut skeletons);
        skeletons
    }
}

/// One concrete operator's propositional effects that have no precondition
/// on their variable, which the cross product ranges over.
struct PropositionalCrossProduct<'a> {
    effects_without_pre: &'a [ExplicitFact],
    has_assignment_effects: bool,
    layout: &'a StateLayout,
}

/// The three fact lists the recursive frames of a cross product share:
/// pushed on the way down and popped on the way back up.
#[derive(Default)]
struct PropositionalFactScratch {
    prev_pairs: Vec<ExplicitFact>,
    pre_pairs: Vec<ExplicitFact>,
    eff_pairs: Vec<ExplicitFact>,
}

impl PropositionalCrossProduct<'_> {
    /// Pushes the skeletons of every value of the effects from `pos` on:
    /// an effect setting `v` to `eff` changes every other value of `v` and
    /// keeps `eff` as a prevail condition.
    fn multiply_out(
        &self,
        pos: usize,
        facts: &mut PropositionalFactScratch,
        skeletons: &mut Vec<PropositionalSkeleton>,
    ) {
        let Some(&effect) = self.effects_without_pre.get(pos) else {
            self.push_skeleton(facts, skeletons);
            return;
        };
        let namespace = effect.namespace();
        let var_id = effect.var();
        let eff = effect.value();
        for value in 0..self.layout.radix(var_id) {
            if value != eff {
                facts
                    .pre_pairs
                    .push(ExplicitFact::in_namespace(namespace, var_id, value));
                facts
                    .eff_pairs
                    .push(ExplicitFact::in_namespace(namespace, var_id, eff));
            } else {
                facts
                    .prev_pairs
                    .push(ExplicitFact::in_namespace(namespace, var_id, value));
            }

            self.multiply_out(pos + 1, facts, skeletons);

            if value != eff {
                facts.pre_pairs.pop();
                facts.eff_pairs.pop();
            } else {
                facts.prev_pairs.pop();
            }
        }
    }

    /// Pushes the skeleton of a full assignment of the cross product, unless
    /// it changes nothing or contradicts itself.
    fn push_skeleton(
        &self,
        facts: &PropositionalFactScratch,
        skeletons: &mut Vec<PropositionalSkeleton>,
    ) {
        if facts.eff_pairs.is_empty() && !self.has_assignment_effects {
            return;
        }
        let normalized = |pairs: &[ExplicitFact]| {
            let mut pairs = pairs.to_vec();
            pairs.sort();
            pairs.dedup();
            pairs
        };
        let pre_pairs = normalized(&facts.pre_pairs);
        let eff_pairs = normalized(&facts.eff_pairs);
        let prev_pairs = normalized(&facts.prev_pairs);
        let has_in_list_conflict = |pairs: &[ExplicitFact]| {
            pairs
                .windows(2)
                .any(|w| w[0].var() == w[1].var() && w[0].value() != w[1].value())
        };
        if has_in_list_conflict(&pre_pairs)
            || has_in_list_conflict(&eff_pairs)
            || has_in_list_conflict(&prev_pairs)
            || has_cross_list_conflict(&pre_pairs, &prev_pairs)
            || has_cross_list_conflict(&prev_pairs, &eff_pairs)
            || !transition_vars_match(&pre_pairs, &eff_pairs)
        {
            return;
        }
        skeletons.push(PropositionalSkeleton {
            prev_pairs,
            pre_pairs,
            eff_pairs,
        });
    }
}

/// Whether the two var-sorted lists `a` and `b` give some variable different
/// values.
fn has_cross_list_conflict(a: &[ExplicitFact], b: &[ExplicitFact]) -> bool {
    let mut i = 0;
    let mut j = 0;
    while i < a.len() && j < b.len() {
        let va = a[i].var();
        let vb = b[j].var();
        if va < vb {
            i += 1;
        } else if vb < va {
            j += 1;
        } else {
            if a[i].value() != b[j].value() {
                return true;
            }
            i += 1;
            j += 1;
        }
    }
    false
}
