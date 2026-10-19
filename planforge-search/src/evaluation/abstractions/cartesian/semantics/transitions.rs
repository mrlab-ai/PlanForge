//! Whether a concrete operator may induce an abstract transition, and the
//! source sets of the transitions it induces.

use super::*;

impl CartesianSemantics<'_> {
    pub(in super::super) fn operator_depends_on_split(
        &self,
        op_id: OperatorIndex,
        dimension: SplitDimension,
    ) -> bool {
        self.split_dependent_operators(dimension).contains(op_id)
    }

    pub(in super::super) fn split_dependent_operators(
        &self,
        dimension: SplitDimension,
    ) -> &OperatorBitSet {
        match dimension {
            SplitDimension::Propositional(var_id) => {
                &self.prop_split_dependent_operators[var_id.index()]
            }
            SplitDimension::Numeric(var_id) => {
                &self.numeric_split_dependent_operators[var_id.index()]
            }
        }
    }

    pub(in super::super) fn invariant_split_dimension_overlaps(
        &self,
        source: &CartesianSet,
        target: &CartesianSet,
        dimension: SplitDimension,
    ) -> bool {
        match dimension {
            SplitDimension::Propositional(var_id) => sorted_value_sets_overlap(
                &source.propositions()[var_id.index()],
                &target.propositions()[var_id.index()],
            ),
            SplitDimension::Numeric(var_id) => {
                source.numeric[var_id.index()].intersects(&target.numeric[var_id.index()])
            }
        }
    }

    pub(in super::super) fn may_transition_after_independent_split(
        &self,
        source: &CartesianSet,
        op_id: OperatorIndex,
        target: &CartesianSet,
        dimension: SplitDimension,
    ) -> Result<bool> {
        debug_assert!(!self.operator_depends_on_split(op_id, dimension));
        let result = self.invariant_split_dimension_overlaps(source, target, dimension);
        #[cfg(debug_assertions)]
        assert_eq!(
            result,
            self.may_transition(source, op_id, target)?,
            "Cartesian split-dependency routing disagrees with full transition semantics for operator {op_id} and dimension {dimension:?}"
        );
        Ok(result)
    }

    pub(in super::super) fn operator_may_apply(
        &self,
        source: &CartesianSet,
        op_id: OperatorIndex,
    ) -> Result<bool> {
        let op = self
            .task
            .get_operators()
            .get(op_id.index())
            .with_context(|| format!("missing operator {op_id}"))?;
        for fact in op.preconditions() {
            if !self.region_admits_fact(source, fact)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    pub(in super::super) fn propositional_dimension_may_transition(
        &self,
        source: &CartesianSet,
        op_id: OperatorIndex,
        target: &CartesianSet,
        var_id: VariableIndex,
    ) -> bool {
        debug_assert!(
            !self.task.numeric_conditions().is_condition_var(var_id)
                && self.propositional_axioms_by_prop_var[var_id.index()].is_empty(),
            "derived proposition {var_id} has no explicit transition relation"
        );
        let op = &self.task.get_operators()[op_id.index()];
        if let Some(effect) = op.effects().iter().find(|effect| effect.var_id() == var_id) {
            debug_assert!(
                effect.conditions().is_empty(),
                "validated Cartesian operator {op_id} has a conditional effect"
            );
            return target.propositions()[var_id.index()]
                .binary_search(&(effect.value() as ExplicitValueIndex))
                .is_ok();
        }
        sorted_value_sets_overlap(
            &source.propositions()[var_id.index()],
            &target.propositions()[var_id.index()],
        )
    }

    pub(in super::super) fn split_dimension_may_transition(
        &self,
        source: &CartesianSet,
        op_id: OperatorIndex,
        target: &CartesianSet,
        dimension: SplitDimension,
    ) -> Result<bool> {
        Ok(match dimension {
            SplitDimension::Propositional(var_id) => {
                self.propositional_dimension_may_transition(source, op_id, target, var_id)
            }
            SplitDimension::Numeric(var_id) => self.numeric_dimension_may_transition(
                source.numeric[var_id.index()],
                op_id,
                target.numeric[var_id.index()],
                var_id,
            )?,
        })
    }

    pub(in super::super) fn numeric_dimension_may_transition(
        &self,
        source: Interval,
        op_id: OperatorIndex,
        target: Interval,
        var_id: VariableIndex,
    ) -> Result<bool> {
        let Some(preimage) = self.numeric_effect_preimage(target, op_id, var_id)? else {
            return Ok(false);
        };
        Ok(!source.is_empty() && preimage.intersects(&source))
    }

    pub(in super::super) fn parent_loop_source_to_split_children(
        &self,
        source: &CartesianSet,
        op_id: OperatorIndex,
        targets: [&CartesianSet; 2],
        dimension: SplitDimension,
    ) -> Result<[bool; 2]> {
        let may_apply = self.operator_may_apply(source, op_id)?;
        let mut result = [false; 2];
        if may_apply {
            for (index, target) in targets.into_iter().enumerate() {
                result[index] =
                    self.split_dimension_may_transition(source, op_id, target, dimension)?;
            }
        }
        #[cfg(debug_assertions)]
        for (index, target) in targets.into_iter().enumerate() {
            assert_eq!(
                result[index],
                self.may_transition(source, op_id, target)?,
                "split-dimension routing disagrees with full transition semantics for parent-loop operator {op_id} and dimension {dimension:?}"
            );
        }
        Ok(result)
    }

    pub(in super::super) fn may_transition(
        &self,
        source: &CartesianSet,
        op_id: OperatorIndex,
        target: &CartesianSet,
    ) -> Result<bool> {
        if !self.operator_may_apply(source, op_id)? {
            return Ok(false);
        }
        for var_id in 0..self.task.get_num_variables() {
            if self
                .task
                .numeric_conditions()
                .is_condition_var(VariableIndex::from_usize(var_id))
                || !self.propositional_axioms_by_prop_var[var_id].is_empty()
            {
                continue;
            }
            if !self.propositional_dimension_may_transition(
                source,
                op_id,
                target,
                VariableIndex::from_usize(var_id),
            ) {
                return Ok(false);
            }
        }

        for (numeric_var_id, variable) in self.task.numeric_variables().iter().enumerate() {
            match variable.get_type() {
                NumericType::Constant => {
                    if !source.numeric[numeric_var_id].intersects(&target.numeric[numeric_var_id]) {
                        return Ok(false);
                    }
                }
                NumericType::Regular => {
                    if !self.numeric_dimension_may_transition(
                        source.numeric[numeric_var_id],
                        op_id,
                        target.numeric[numeric_var_id],
                        VariableIndex::from_usize(numeric_var_id),
                    )? {
                        return Ok(false);
                    }
                }
                NumericType::Derived => {
                    if self.coordinates.views()[numeric_var_id].is_some()
                        && !self.numeric_dimension_may_transition(
                            source.numeric[numeric_var_id],
                            op_id,
                            target.numeric[numeric_var_id],
                            VariableIndex::from_usize(numeric_var_id),
                        )?
                    {
                        return Ok(false);
                    }
                }
                NumericType::Cost => {}
            }
        }
        Ok(true)
    }

    pub(in super::super) fn numeric_effect_preimage(
        &self,
        target: Interval,
        op_id: OperatorIndex,
        numeric_var_id: VariableIndex,
    ) -> Result<Option<Interval>> {
        let mut preimage = target;
        if let Some(view) = self.coordinates.views()[numeric_var_id.index()].as_ref() {
            let delta = float_tolerance::canonicalize(view.operator_delta(op_id)?);
            preimage.apply_reverse_op(&AssignmentOperation::Plus, &Interval::singleton(delta));
            return Ok(Some(preimage));
        }
        let op = &self.task.get_operators()[op_id.index()];
        for effect in op
            .assignment_effects()
            .iter()
            .filter(|effect| effect.affected_var_id() == numeric_var_id)
            .rev()
        {
            let rhs = float_tolerance::canonicalize(
                self.task.get_initial_numeric_state_values()[effect.var_id().index()],
            );
            match effect.operation() {
                AssignmentOperation::Assign => {
                    if !preimage.contains(rhs) {
                        return Ok(None);
                    }
                    preimage = Interval::unbounded();
                }
                AssignmentOperation::Plus => {
                    preimage
                        .apply_reverse_op(&AssignmentOperation::Plus, &Interval::singleton(rhs));
                }
                AssignmentOperation::Minus => {
                    preimage
                        .apply_reverse_op(&AssignmentOperation::Minus, &Interval::singleton(rhs));
                }
                AssignmentOperation::Times => {
                    if rhs == ZERO_VALUE {
                        if !preimage.contains(ZERO_VALUE) {
                            return Ok(None);
                        }
                        preimage = Interval::unbounded();
                    } else {
                        preimage.apply_reverse_op(
                            &AssignmentOperation::Times,
                            &Interval::singleton(rhs),
                        );
                    }
                }
                AssignmentOperation::Divide => {
                    preimage
                        .apply_reverse_op(&AssignmentOperation::Divide, &Interval::singleton(rhs));
                }
            }
            preimage = preimage.canonicalized();
        }
        Ok(Some(preimage))
    }

    pub(in super::super) fn source_set_source_for_transition(
        &self,
        source: &CartesianSet,
        op_id: OperatorIndex,
        target: &CartesianSet,
    ) -> Result<Option<CartesianSet>> {
        debug_assert_eq!(
            source.numeric.len(),
            target.numeric.len(),
            "Cartesian transition source/target numeric dimension mismatch"
        );
        let mut source_set = source.clone();
        for (numeric_var_id, variable) in self.task.numeric_variables().iter().enumerate() {
            match variable.get_type() {
                NumericType::Constant => {
                    if !source.numeric[numeric_var_id].intersects(&target.numeric[numeric_var_id]) {
                        return Ok(None);
                    }
                }
                NumericType::Regular => {
                    let Some(preimage) = self.numeric_effect_preimage(
                        target.numeric[numeric_var_id],
                        op_id,
                        VariableIndex::from_usize(numeric_var_id),
                    )?
                    else {
                        return Ok(None);
                    };
                    let regressed = source.numeric[numeric_var_id].intersection(&preimage);
                    if regressed.is_empty() {
                        return Ok(None);
                    }
                    if regressed != source.numeric[numeric_var_id] {
                        Arc::make_mut(&mut source_set.numeric)[numeric_var_id] = regressed;
                    }
                }
                NumericType::Derived => {
                    if self.coordinates.views()[numeric_var_id].is_none() {
                        continue;
                    }
                    let preimage = self
                        .numeric_effect_preimage(
                            target.numeric[numeric_var_id],
                            op_id,
                            VariableIndex::from_usize(numeric_var_id),
                        )?
                        .expect("additive-view preimage is always defined");
                    let regressed = source.numeric[numeric_var_id].intersection(&preimage);
                    if regressed.is_empty() {
                        return Ok(None);
                    }
                    if regressed != source.numeric[numeric_var_id] {
                        Arc::make_mut(&mut source_set.numeric)[numeric_var_id] = regressed;
                    }
                }
                NumericType::Cost => {}
            }
        }
        Ok(Some(source_set))
    }
}
