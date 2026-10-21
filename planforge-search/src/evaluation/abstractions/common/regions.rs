//! What a task's facts and operators mean for Cartesian sets of its states.
//!
//! Abstract states of Cartesian and domain abstractions are Cartesian sets:
//! per propositional variable a set of values, per numeric coordinate an
//! interval. A region *admits* a fact if some of its states satisfy it and
//! *guarantees* it if all do. Conditions are derived from the numeric
//! coordinates: a comparison holds in a region where its interval evaluation
//! says so. Propositional axioms are evaluated through the axioms that
//! support a value, and their default value holds where no supporting axiom
//! is guaranteed to fire.

#[cfg(test)]
mod tests;

use std::sync::Arc;

use anyhow::{Context, Result, ensure};
use planforge_sas::numeric_conditions::ConditionValue;
use planforge_sas::numeric_task::{
    AbstractNumericTask, ExplicitFact, ExplicitValueIndex, NumericType, OperatorIndex,
    VariableIndex,
};
use planforge_sas::utils::float_tolerance;
use planforge_sas::utils::interval::Interval;

use super::cartesian_set::{CartesianSet, cartesian_set_is_nonempty, sorted_value_sets_overlap};
use super::coordinates::Coordinates;
use super::transition_system::{AbstractOperatorSourceSets, LabelSourceSet};

/// The interval of numeric variable `var_id` of `task` in the trivial
/// abstraction: a constant at its value, every other variable unbounded.
pub(crate) fn trivial_interval(task: &dyn AbstractNumericTask, var_id: VariableIndex) -> Interval {
    let variable = &task.numeric_variables()[var_id.index()];
    if variable.get_type() != &NumericType::Constant {
        return Interval::unbounded();
    }
    let value =
        float_tolerance::canonicalize(task.get_initial_numeric_state_values()[var_id.index()]);
    assert!(
        value.is_finite(),
        "constant numeric variable {} has the non-finite value {value}",
        variable.name()
    );
    Interval::singleton(value)
}

/// The deadline operation of building source sets.
pub(crate) const SOURCE_SETS: &str = "abstract source sets";

/// What regions of the concrete state space admit, guarantee, and let an
/// operator lead into.
pub struct RegionSemantics<'task> {
    task: &'task dyn AbstractNumericTask,
    coordinates: Coordinates<'task>,
    /// The propositional axioms writing each propositional variable.
    axioms_by_var: Vec<Vec<usize>>,
    /// Per propositional variable with axioms, the value it has where none
    /// of them fires.
    axiom_defaults: Vec<Option<ExplicitValueIndex>>,
    /// Per operator, the value its successors have on each operator
    /// variable it writes or requires, sorted by variable: the effect's
    /// value, else the prevail condition's.
    successor_values: Vec<Box<[(VariableIndex, ExplicitValueIndex)]>>,
}

impl<'task> RegionSemantics<'task> {
    pub(crate) fn new(task: &'task dyn AbstractNumericTask) -> Result<Self> {
        let mut axioms_by_var = vec![Vec::new(); task.get_num_variables()];
        for (axiom_id, axiom) in task.axioms().iter().enumerate() {
            let var_id = axiom.var_id();
            ensure!(
                var_id.index() < axioms_by_var.len(),
                "propositional axiom {axiom_id} affects missing prop var {var_id}"
            );
            axioms_by_var[var_id.index()].push(axiom_id);
        }
        let axiom_defaults = axioms_by_var
            .iter()
            .enumerate()
            .map(|(var_id, axiom_ids)| {
                let Some((&first_axiom_id, remaining)) = axiom_ids.split_first() else {
                    return Ok(None);
                };
                let default_value = task.axioms()[first_axiom_id].precondition_value();
                ensure!(
                    remaining
                        .iter()
                        .all(|&axiom_id| task.axioms()[axiom_id].precondition_value()
                            == default_value),
                    "propositional axioms for variable {var_id} disagree on default value"
                );
                Ok(Some(default_value))
            })
            .collect::<Result<_>>()?;
        let is_operator_variable = |var_id: VariableIndex| {
            !task.numeric_conditions().is_condition_var(var_id)
                && axioms_by_var[var_id.index()].is_empty()
        };
        let successor_values = task
            .get_operators()
            .iter()
            .enumerate()
            .map(|(op_id, op)| {
                // Effects before prevail conditions, each in order: the first
                // entry of a variable is the one that determines its value.
                let mut values =
                    op.effects()
                        .iter()
                        .map(|effect| {
                            debug_assert!(
                                effect.conditions().is_empty(),
                                "validated operator {op_id} has a conditional effect"
                            );
                            (effect.var_id(), effect.value() as ExplicitValueIndex)
                        })
                        .chain(op.preconditions().iter().map(|precondition| {
                            (precondition.var_index(), precondition.value_index())
                        }))
                        .filter(|&(var_id, _)| is_operator_variable(var_id))
                        .collect::<Vec<_>>();
                values.sort_by_key(|&(var_id, _)| var_id);
                values.dedup_by_key(|&mut (var_id, _)| var_id);
                values.into_boxed_slice()
            })
            .collect();
        Ok(Self {
            task,
            coordinates: Coordinates::new(task),
            axioms_by_var,
            axiom_defaults,
            successor_values,
        })
    }

    pub(crate) fn task(&self) -> &'task dyn AbstractNumericTask {
        self.task
    }

    pub(crate) fn coordinates(&self) -> &Coordinates<'task> {
        &self.coordinates
    }

    /// The propositional axioms writing `var_id`; empty for a variable that
    /// operators write.
    pub(crate) fn axioms_for(&self, var_id: VariableIndex) -> &[usize] {
        &self.axioms_by_var[var_id.index()]
    }

    /// Whether `var_id` is written by operators, rather than derived by a
    /// comparison or by propositional axioms.
    pub(crate) fn is_operator_variable(&self, var_id: VariableIndex) -> bool {
        !self.task.numeric_conditions().is_condition_var(var_id)
            && self.axioms_by_var[var_id.index()].is_empty()
    }

    /// The region of every state: all propositional values, unbounded
    /// numeric coordinates, constants at their values.
    pub(crate) fn trivial_region(&self) -> Result<CartesianSet> {
        let propositions = (0..self.task.get_num_variables())
            .map(|var_id| {
                let size = self.task.variables()[var_id].domain_size();
                ensure!(size > 0, "propositional var {var_id} has an empty domain");
                ensure!(
                    u32::try_from(size).is_ok(),
                    "propositional var {var_id} domain is too large: {size}"
                );
                Ok((0..size as u32).map(ExplicitValueIndex::new).collect())
            })
            .collect::<Result<Vec<_>>>()?;
        let numeric: Vec<_> = (0..self.task.numeric_variables().len())
            .map(|var_id| trivial_interval(self.task, VariableIndex::from_usize(var_id)))
            .collect();
        // Every dimension carries its whole domain, so nothing is constrained
        // yet; refinement adds constraints one dimension at a time.
        Ok(CartesianSet::with_constrained_props(
            propositions,
            numeric,
            Arc::from([]),
        ))
    }

    /// Whether some state of `region` satisfies `fact`.
    pub(crate) fn admits(&self, region: &CartesianSet, fact: &ExplicitFact) -> Result<bool> {
        self.admits_with(region, fact, &mut Vec::new())
    }

    /// Whether every state of `region` satisfies `fact`.
    pub(crate) fn guarantees(&self, region: &CartesianSet, fact: &ExplicitFact) -> Result<bool> {
        self.guarantees_with(region, fact, &mut Vec::new())
    }

    /// Whether `region` admits each of `facts` (not necessarily in one state).
    pub(crate) fn admits_all(&self, region: &CartesianSet, facts: &[ExplicitFact]) -> Result<bool> {
        self.admits_all_with(region, facts, &mut Vec::new())
    }

    /// Whether `region` admits every goal fact.
    pub(crate) fn is_goal(&self, region: &CartesianSet) -> Result<bool> {
        for goal in self.task.goals() {
            if !self.admits(region, goal)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// The value a propositional axiom variable has where none of its axioms
    /// fires.
    pub(crate) fn axiom_default(&self, var_id: VariableIndex) -> ExplicitValueIndex {
        self.axiom_defaults[var_id.index()]
            .unwrap_or_else(|| panic!("variable {var_id} has no propositional axiom"))
    }

    /// Whether comparison `tree_id` may be true and may be false in `region`.
    pub(crate) fn comparison_truths(
        &self,
        region: &CartesianSet,
        tree_id: usize,
    ) -> Result<(bool, bool)> {
        let tree = self
            .task
            .numeric_conditions()
            .get(tree_id)
            .with_context(|| format!("missing comparison tree {tree_id}"))?;
        ensure!(
            region.numeric.iter().all(|interval| !interval.is_empty()),
            "comparison tree {tree_id} evaluated on an empty region"
        );
        Ok(match tree.evaluate_interval(&region.numeric) {
            Some(true) => (true, false),
            Some(false) => (false, true),
            None => (true, true),
        })
    }

    /// Whether some state of `source` satisfies every precondition of `op_id`.
    pub(crate) fn operator_may_apply(
        &self,
        source: &CartesianSet,
        op_id: OperatorIndex,
    ) -> Result<bool> {
        let op = self
            .task
            .get_operators()
            .get(op_id.index())
            .with_context(|| format!("missing operator {op_id}"))?;
        self.admits_all(source, op.preconditions())
    }

    /// Whether applying `op_id` in `source` may lead to `target` on the
    /// operator variable `var_id`: the successor has the effect's value, or
    /// a prevail condition's value, or keeps a value of `source`.
    pub(crate) fn propositional_may_transition(
        &self,
        source: &CartesianSet,
        op_id: OperatorIndex,
        target: &CartesianSet,
        var_id: VariableIndex,
    ) -> bool {
        debug_assert!(
            self.is_operator_variable(var_id),
            "derived proposition {var_id} has no explicit transition relation"
        );
        let successor_values = &self.successor_values[op_id.index()];
        let target_values = &target.propositions()[var_id.index()];
        match successor_values.binary_search_by_key(&var_id, |&(var_id, _)| var_id) {
            Ok(index) => target_values
                .binary_search(&successor_values[index].1)
                .is_ok(),
            Err(_) => {
                sorted_value_sets_overlap(&source.propositions()[var_id.index()], target_values)
            }
        }
    }

    /// Whether applying `op_id` with numeric coordinate `var_id` in `source`
    /// may leave it in `target`.
    pub(crate) fn numeric_may_transition(
        &self,
        source: Interval,
        op_id: OperatorIndex,
        target: Interval,
        var_id: VariableIndex,
    ) -> Result<bool> {
        let Some(preimage) = self.coordinates.preimage(target, op_id, var_id) else {
            return Ok(false);
        };
        Ok(!source.is_empty() && preimage.intersects(&source))
    }

    /// Whether `op_id` may induce an abstract transition from `source` to
    /// `target`: it may apply in `source`, and on every operator variable and
    /// numeric coordinate some applicable state may reach `target`. A
    /// variable `target` does not constrain admits every successor value.
    pub(crate) fn may_transition(
        &self,
        source: &CartesianSet,
        op_id: OperatorIndex,
        target: &CartesianSet,
    ) -> Result<bool> {
        debug_assert!(
            cartesian_set_is_nonempty(source),
            "transition source region is empty"
        );
        if !self.operator_may_apply(source, op_id)? {
            return Ok(false);
        }
        for var_id in target
            .constrained_props()
            .iter()
            .map(|&var_id| VariableIndex::new(var_id))
        {
            if self.is_operator_variable(var_id)
                && !self.propositional_may_transition(source, op_id, target, var_id)
            {
                return Ok(false);
            }
        }
        for (numeric_var_id, variable) in self.task.numeric_variables().iter().enumerate() {
            let var_id = VariableIndex::from_usize(numeric_var_id);
            let may_transition = match variable.get_type() {
                NumericType::Constant => {
                    source.numeric[numeric_var_id].intersects(&target.numeric[numeric_var_id])
                }
                NumericType::Regular => self.numeric_may_transition(
                    source.numeric[numeric_var_id],
                    op_id,
                    target.numeric[numeric_var_id],
                    var_id,
                )?,
                NumericType::Derived if self.coordinates.views()[numeric_var_id].is_some() => self
                    .numeric_may_transition(
                        source.numeric[numeric_var_id],
                        op_id,
                        target.numeric[numeric_var_id],
                        var_id,
                    )?,
                NumericType::Derived | NumericType::Cost => true,
            };
            if !may_transition {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// The states of `source` from which `op_id` may lead into `target`:
    /// `source` narrowed on every numeric coordinate to the preimage of
    /// `target`, or `None` if that is empty.
    pub(crate) fn transition_source_set(
        &self,
        source: &CartesianSet,
        op_id: OperatorIndex,
        target: &CartesianSet,
    ) -> Result<Option<CartesianSet>> {
        debug_assert_eq!(
            source.numeric.len(),
            target.numeric.len(),
            "transition source/target numeric dimension mismatch"
        );
        let mut source_set = source.clone();
        for (numeric_var_id, variable) in self.task.numeric_variables().iter().enumerate() {
            let var_id = VariableIndex::from_usize(numeric_var_id);
            let preimage = match variable.get_type() {
                NumericType::Constant => {
                    if !source.numeric[numeric_var_id].intersects(&target.numeric[numeric_var_id]) {
                        return Ok(None);
                    }
                    continue;
                }
                NumericType::Regular => {
                    let Some(preimage) =
                        self.coordinates
                            .preimage(target.numeric[numeric_var_id], op_id, var_id)
                    else {
                        return Ok(None);
                    };
                    preimage
                }
                NumericType::Derived if self.coordinates.views()[numeric_var_id].is_some() => self
                    .coordinates
                    .preimage(target.numeric[numeric_var_id], op_id, var_id)
                    .expect("additive-view preimage is always defined"),
                NumericType::Derived | NumericType::Cost => continue,
            };
            let regressed = source.numeric[numeric_var_id].intersection(&preimage);
            if regressed.is_empty() {
                return Ok(None);
            }
            if regressed != source.numeric[numeric_var_id] {
                Arc::make_mut(&mut source_set.numeric)[numeric_var_id] = regressed;
            }
        }
        Ok(Some(source_set))
    }

    /// The source sets of an abstract operator from `source` to `target`
    /// with `labels`: for each label, the states of `source` from which it
    /// may lead into `target`. A label that may leave from all of `source`
    /// shares it.
    pub(crate) fn abstract_operator_source_sets(
        &self,
        source: &Arc<CartesianSet>,
        labels: &[OperatorIndex],
        target: &CartesianSet,
    ) -> Result<AbstractOperatorSourceSets> {
        let labels = labels
            .iter()
            .map(|&concrete_op_id| {
                let source_set = self
                    .transition_source_set(source, concrete_op_id, target)?
                    .with_context(|| {
                        format!("operator {concrete_op_id} has an empty source set")
                    })?;
                let source = if source_set == **source {
                    Arc::clone(source)
                } else {
                    Arc::new(source_set)
                };
                Ok(LabelSourceSet {
                    concrete_op_id,
                    source,
                })
            })
            .collect::<Result<_>>()?;
        Ok(AbstractOperatorSourceSets { labels })
    }

    fn admits_with(
        &self,
        region: &CartesianSet,
        fact: &ExplicitFact,
        visiting: &mut Vec<VariableIndex>,
    ) -> Result<bool> {
        let var_id = fact.var_index();
        if let Some(tree_id) = self.task.numeric_conditions().id_for_var(var_id) {
            let (may_true, may_false) = self.comparison_truths(region, tree_id)?;
            return Ok(match comparison_value(fact)? {
                ConditionValue::True => may_true,
                ConditionValue::False => may_false,
            });
        }
        if !self.axioms_for(var_id).is_empty() {
            return self.with_visiting(var_id, visiting, |visiting| {
                let default_value = self.axiom_default(var_id);
                if fact.value_index() == default_value {
                    for &axiom_id in self.axioms_for(var_id) {
                        let conditions = self.task.axioms()[axiom_id].conditions();
                        if self.guarantees_all_with(region, conditions, visiting)? {
                            return Ok(false);
                        }
                    }
                    return Ok(true);
                }
                for &axiom_id in self.axioms_for(var_id) {
                    let axiom = &self.task.axioms()[axiom_id];
                    if axiom.effect_value() == fact.value_index()
                        && self.admits_all_with(region, axiom.conditions(), visiting)?
                    {
                        return Ok(true);
                    }
                }
                Ok(false)
            });
        }
        Ok(region.propositions()[var_id.index()]
            .binary_search(&fact.value_index())
            .is_ok())
    }

    fn guarantees_with(
        &self,
        region: &CartesianSet,
        fact: &ExplicitFact,
        visiting: &mut Vec<VariableIndex>,
    ) -> Result<bool> {
        let var_id = fact.var_index();
        if let Some(tree_id) = self.task.numeric_conditions().id_for_var(var_id) {
            let (may_true, may_false) = self.comparison_truths(region, tree_id)?;
            return Ok(match comparison_value(fact)? {
                ConditionValue::True => may_true && !may_false,
                ConditionValue::False => may_false && !may_true,
            });
        }
        if !self.axioms_for(var_id).is_empty() {
            return self.with_visiting(var_id, visiting, |visiting| {
                let default_value = self.axiom_default(var_id);
                if fact.value_index() == default_value {
                    for &axiom_id in self.axioms_for(var_id) {
                        let conditions = self.task.axioms()[axiom_id].conditions();
                        if self.admits_all_with(region, conditions, visiting)? {
                            return Ok(false);
                        }
                    }
                    return Ok(true);
                }
                for &axiom_id in self.axioms_for(var_id) {
                    let axiom = &self.task.axioms()[axiom_id];
                    if axiom.effect_value() == fact.value_index()
                        && self.guarantees_all_with(region, axiom.conditions(), visiting)?
                    {
                        return Ok(true);
                    }
                }
                Ok(false)
            });
        }
        let values = &region.propositions()[var_id.index()];
        Ok(values.len() == 1 && values[0] == fact.value_index())
    }

    fn admits_all_with(
        &self,
        region: &CartesianSet,
        facts: &[ExplicitFact],
        visiting: &mut Vec<VariableIndex>,
    ) -> Result<bool> {
        for fact in facts {
            if !self.admits_with(region, fact, visiting)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn guarantees_all_with(
        &self,
        region: &CartesianSet,
        facts: &[ExplicitFact],
        visiting: &mut Vec<VariableIndex>,
    ) -> Result<bool> {
        for fact in facts {
            if !self.guarantees_with(region, fact, visiting)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Evaluate `evaluate` with `var_id` on the stack of the derived
    /// variables being evaluated, `visiting`, rejecting cyclic axiom support.
    fn with_visiting(
        &self,
        var_id: VariableIndex,
        visiting: &mut Vec<VariableIndex>,
        evaluate: impl FnOnce(&mut Vec<VariableIndex>) -> Result<bool>,
    ) -> Result<bool> {
        ensure!(
            !visiting.contains(&var_id),
            "cyclic propositional axiom support for variable {var_id}"
        );
        visiting.push(var_id);
        let result = evaluate(visiting);
        visiting.pop();
        result
    }
}

/// The truth value the comparison fact `fact` asks for.
fn comparison_value(fact: &ExplicitFact) -> Result<ConditionValue> {
    ConditionValue::from_usize(fact.value()).with_context(|| {
        format!(
            "invalid comparison proposition value {} for var {}",
            fact.value(),
            fact.var()
        )
    })
}
