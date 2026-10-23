//! Which facts a Cartesian region admits or guarantees.

use super::*;

impl CartesianSemantics<'_> {
    pub(in super::super) fn trivial_region(&self) -> Result<CartesianSet> {
        let propositions = (0..self.task.get_num_variables())
            .map(|var_id| {
                let size = self
                    .task
                    .get_variable_domain_size(VariableIndex::from_usize(var_id))
                    .map_err(|error| anyhow::anyhow!(error.to_string()))?;
                ensure!(size > 0, "propositional var {var_id} has an empty domain");
                ensure!(
                    u32::try_from(size).is_ok(),
                    "propositional var {var_id} domain is too large: {size}"
                );
                Ok((0..size as u32).map(ExplicitValueIndex::new).collect())
            })
            .collect::<Result<Vec<_>>>()?;
        let initial_numeric = self.task.get_initial_numeric_state_values();
        let numeric: Vec<_> = self
            .task
            .numeric_variables()
            .iter()
            .enumerate()
            .map(|(var_id, var)| {
                if matches!(var.get_type(), NumericType::Constant) {
                    Interval::singleton(initial_numeric[var_id])
                } else {
                    Interval::unbounded()
                }
            })
            .collect();
        // Every dimension carries its whole domain, so nothing is constrained
        // yet; refinement splits add constraints one dimension at a time.
        Ok(CartesianSet::with_constrained_props(
            propositions,
            numeric,
            Arc::from([]),
        ))
    }

    pub(in super::super) fn region_admits_fact(
        &self,
        region: &CartesianSet,
        fact: &ExplicitFact,
    ) -> Result<bool> {
        let mut visiting = vec![false; self.task.get_num_variables()];
        self.region_admits_fact_inner(region, fact, &mut visiting)
    }

    pub(in super::super) fn region_admits_fact_inner(
        &self,
        region: &CartesianSet,
        fact: &ExplicitFact,
        visiting: &mut [bool],
    ) -> Result<bool> {
        let var_id = fact.var_index();
        if let Some(axiom_id) = self.task.numeric_conditions().id_for_var(var_id) {
            let (may_true, may_false) = self.comparison_truths(region, axiom_id)?;
            return Ok(match fact.value() {
                0 => may_true,
                1 => may_false,
                2 => may_true || may_false,
                value => bail!("invalid comparison proposition value {value} for var {var_id}"),
            });
        }
        if !self.propositional_axioms_by_prop_var[var_id.index()].is_empty() {
            ensure!(
                !visiting[var_id.index()],
                "cyclic propositional axiom support for variable {var_id}"
            );
            visiting[var_id.index()] = true;
            let result = (|| {
                let default_value = self.propositional_axiom_default(var_id.index())?;
                if fact.value_index() == default_value {
                    for &axiom_id in &self.propositional_axioms_by_prop_var[var_id.index()] {
                        let axiom = &self.task.axioms()[axiom_id];
                        if self.all_conditions_guaranteed(region, axiom.conditions(), visiting)? {
                            return Ok(false);
                        }
                    }
                    return Ok(true);
                }

                for &axiom_id in &self.propositional_axioms_by_prop_var[var_id.index()] {
                    let axiom = &self.task.axioms()[axiom_id];
                    if axiom.effect_value() == fact.value_index()
                        && self.all_conditions_admitted(region, axiom.conditions(), visiting)?
                    {
                        return Ok(true);
                    }
                }
                Ok(false)
            })();
            visiting[var_id.index()] = false;
            return result;
        }
        Ok(region
            .propositions()
            .get(var_id.index())
            .is_some_and(|values| values.binary_search(&(fact.value_index())).is_ok()))
    }

    pub(in super::super) fn region_guarantees_fact(
        &self,
        region: &CartesianSet,
        fact: &ExplicitFact,
    ) -> Result<bool> {
        let mut visiting = vec![false; self.task.get_num_variables()];
        self.region_guarantees_fact_inner(region, fact, &mut visiting)
    }

    pub(in super::super) fn region_guarantees_fact_inner(
        &self,
        region: &CartesianSet,
        fact: &ExplicitFact,
        visiting: &mut [bool],
    ) -> Result<bool> {
        let var_id = fact.var_index();
        if let Some(axiom_id) = self.task.numeric_conditions().id_for_var(var_id) {
            let (may_true, may_false) = self.comparison_truths(region, axiom_id)?;
            return Ok(match fact.value() {
                0 => may_true && !may_false,
                1 => may_false && !may_true,
                2 => false,
                value => bail!("invalid comparison proposition value {value} for var {var_id}"),
            });
        }
        if !self.propositional_axioms_by_prop_var[var_id.index()].is_empty() {
            ensure!(
                !visiting[var_id.index()],
                "cyclic propositional axiom support for variable {var_id}"
            );
            visiting[var_id.index()] = true;
            let result = (|| {
                let default_value = self.propositional_axiom_default(var_id.index())?;
                if fact.value_index() == default_value {
                    for &axiom_id in &self.propositional_axioms_by_prop_var[var_id.index()] {
                        let axiom = &self.task.axioms()[axiom_id];
                        if self.all_conditions_admitted(region, axiom.conditions(), visiting)? {
                            return Ok(false);
                        }
                    }
                    return Ok(true);
                }

                for &axiom_id in &self.propositional_axioms_by_prop_var[var_id.index()] {
                    let axiom = &self.task.axioms()[axiom_id];
                    if axiom.effect_value() == fact.value_index()
                        && self.all_conditions_guaranteed(region, axiom.conditions(), visiting)?
                    {
                        return Ok(true);
                    }
                }
                Ok(false)
            })();
            visiting[var_id.index()] = false;
            return result;
        }
        let Some(values) = region.propositions().get(var_id.index()) else {
            return Ok(false);
        };
        Ok(values.len() == 1 && values[0] == fact.value_index())
    }

    pub(in super::super) fn all_conditions_admitted(
        &self,
        region: &CartesianSet,
        conditions: &[ExplicitFact],
        visiting: &mut [bool],
    ) -> Result<bool> {
        for condition in conditions {
            if !self.region_admits_fact_inner(region, condition, visiting)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    pub(in super::super) fn all_conditions_guaranteed(
        &self,
        region: &CartesianSet,
        conditions: &[ExplicitFact],
        visiting: &mut [bool],
    ) -> Result<bool> {
        for condition in conditions {
            if !self.region_guarantees_fact_inner(region, condition, visiting)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    pub(in super::super) fn propositional_axiom_default(
        &self,
        var_id: usize,
    ) -> Result<ExplicitValueIndex> {
        let axiom_ids = self
            .propositional_axioms_by_prop_var
            .get(var_id)
            .with_context(|| format!("missing propositional variable {var_id}"))?;
        let (&first_axiom_id, remaining) = axiom_ids
            .split_first()
            .with_context(|| format!("variable {var_id} has no propositional axiom"))?;
        let default_value = self.task.axioms()[first_axiom_id].precondition_value();
        for &axiom_id in remaining {
            ensure!(
                self.task.axioms()[axiom_id].precondition_value() == default_value,
                "propositional axioms for variable {var_id} disagree on default value"
            );
        }
        Ok(default_value)
    }

    pub(in super::super) fn comparison_truths(
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
            "comparison tree {tree_id} evaluated on an empty Cartesian region"
        );
        Ok(match tree.evaluate_interval(&region.numeric) {
            Some(true) => (true, false),
            Some(false) => (false, true),
            None => (true, true),
        })
    }

    pub(in super::super) fn region_is_goal(&self, region: &CartesianSet) -> Result<bool> {
        for goal_id in 0..self.task.get_num_goals() {
            if !self.region_admits_fact(region, self.task.get_goal_fact(goal_id))? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    pub(in super::super) fn concrete_prop_values(
        &self,
        packer: &StatePacker,
        packed: &[u64],
        out: &mut Vec<ExplicitValueIndex>,
    ) {
        out.clear();
        out.extend(
            (0..self.task.get_num_variables())
                .map(|var_id| ExplicitValueIndex::from_usize(packer.get(packed, var_id) as usize)),
        );
    }
}
