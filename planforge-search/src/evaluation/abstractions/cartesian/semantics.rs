use super::*;

pub(super) struct CartesianSemantics<'task> {
    pub(super) task: &'task dyn AbstractNumericTask,
    pub(super) propositional_axioms_by_prop_var: Vec<Vec<usize>>,
    pub(super) operator_costs: Vec<NumericValue>,
    pub(super) prop_split_dependent_operators: Vec<OperatorBitSet>,
    pub(super) numeric_split_dependent_operators: Vec<OperatorBitSet>,
    pub(super) coordinates: Coordinates<'task>,
    pub(super) additive_effect_deltas: Vec<Vec<NumericValue>>,
    pub(super) numeric_integer_lattice: Vec<bool>,
    pub(super) random_seed: Option<u64>,
    pub(super) random_split_rng: Option<RefCell<SmallRng>>,
    pub(super) icaps_random: Option<RefCell<ArtifactMt19937>>,
    pub(super) refinement_direction: CartesianRefinementDirection,
    pub(super) flaw_candidate_generation: CartesianFlawCandidateGeneration,
    pub(super) split_selection_rank: Option<usize>,
    pub(super) split_selection: CartesianSplitSelection,
    pub(super) target_split_boundaries: Vec<NumericValue>,
}

pub(super) fn mark_fact_split_dependencies(
    task: &dyn AbstractNumericTask,
    fact: &ExplicitFact,
    propositional_axioms_by_prop_var: &[Vec<usize>],
    visiting: &mut [bool],
    prop_dependencies: &mut [bool],
    numeric_dependencies: &mut [bool],
) -> Result<()> {
    let var_id = fact.var_index();
    if let Some(condition) = task.numeric_conditions().for_var(var_id) {
        let mut dimensions = condition.regular_numeric_var_dependencies().to_vec();
        dimensions.extend(comparison_refinement_dimensions(task, condition));
        dimensions.sort_unstable();
        dimensions.dedup();
        for numeric_var_id in dimensions {
            numeric_dependencies[numeric_var_id.index()] = true;
        }
        return Ok(());
    }
    if propositional_axioms_by_prop_var[var_id.index()].is_empty() {
        prop_dependencies[var_id.index()] = true;
        return Ok(());
    }
    ensure!(
        !visiting[var_id.index()],
        "cyclic propositional axiom dependency through variable {var_id}"
    );
    visiting[var_id.index()] = true;
    for &axiom_id in &propositional_axioms_by_prop_var[var_id.index()] {
        let axiom = task
            .axioms()
            .get(axiom_id)
            .with_context(|| format!("missing propositional axiom {axiom_id}"))?;
        for condition in axiom.conditions() {
            mark_fact_split_dependencies(
                task,
                condition,
                propositional_axioms_by_prop_var,
                visiting,
                prop_dependencies,
                numeric_dependencies,
            )?;
        }
    }
    visiting[var_id.index()] = false;
    Ok(())
}

impl<'task> CartesianSemantics<'task> {
    pub(super) fn task(&self) -> &dyn AbstractNumericTask {
        self.task
    }

    pub(super) fn propositional_axioms_by_prop_var(&self) -> &[Vec<usize>] {
        &self.propositional_axioms_by_prop_var
    }

    pub(super) fn operator_costs(&self) -> &[NumericValue] {
        &self.operator_costs
    }

    pub(super) fn additive_numeric_views(&self) -> &[Option<AdditiveNumericView>] {
        self.coordinates.views()
    }

    pub(super) fn coordinates(&self) -> &Coordinates<'task> {
        &self.coordinates
    }

    pub(super) fn additive_effect_deltas(&self) -> &[Vec<NumericValue>] {
        &self.additive_effect_deltas
    }

    pub(super) fn numeric_integer_lattice(&self) -> &[bool] {
        &self.numeric_integer_lattice
    }

    pub(super) fn refinement_direction(&self) -> CartesianRefinementDirection {
        self.refinement_direction
    }

    pub(super) fn flaw_candidate_generation(&self) -> CartesianFlawCandidateGeneration {
        self.flaw_candidate_generation
    }

    pub(super) fn split_selection(&self) -> CartesianSplitSelection {
        self.split_selection
    }

    pub(super) fn target_split_boundaries(&self) -> &[NumericValue] {
        &self.target_split_boundaries
    }

    pub(super) fn new(
        task: &'task dyn AbstractNumericTask,
        config: &CartesianAbstractionConfig,
    ) -> Result<Self> {
        validate_abstractable_goal(task).map_err(anyhow::Error::msg)?;
        let coordinates = Coordinates::new(task);
        if config.flaw_kind == FlawKind::Regression {
            ensure_regressable(task, &coordinates)?;
        }
        for (op_id, op) in task.get_operators().iter().enumerate() {
            validate_abstraction_operator(task, op, OperatorIndex::from_usize(op_id))?;
        }

        let mut propositional_axioms_by_prop_var = vec![Vec::new(); task.get_num_variables()];
        for (axiom_id, axiom) in task.axioms().iter().enumerate() {
            let var_id = axiom.var_id();
            ensure!(
                var_id.index() < propositional_axioms_by_prop_var.len(),
                "propositional axiom {axiom_id} affects missing prop var {var_id}"
            );
            propositional_axioms_by_prop_var[var_id.index()].push(axiom_id);
        }
        let operator_costs = task
            .get_operators()
            .iter()
            .map(|op| metric_operator_cost_from_initial_values(task, op))
            .collect();
        let operator_count = task.get_operators().len();
        let additive_numeric_views = coordinates.views();
        let mut additive_effect_deltas = Vec::with_capacity(task.numeric_variables().len());
        for (numeric_var_id, variable) in task.numeric_variables().iter().enumerate() {
            let mut deltas = match variable.get_type() {
                NumericType::Regular => task
                    .get_operators()
                    .iter()
                    .filter_map(|operator| {
                        numeric_dimension_delta_for_operator(
                            task,
                            VariableIndex::from_usize(numeric_var_id),
                            operator,
                        )
                    })
                    .collect::<Vec<_>>(),
                NumericType::Derived => additive_numeric_views[numeric_var_id]
                    .as_ref()
                    .map(|view| view.operator_deltas().to_vec())
                    .unwrap_or_default(),
                NumericType::Constant | NumericType::Cost => Vec::new(),
            };
            deltas.retain(|delta| delta.value().abs() > float_tolerance::SEARCH_EPSILON);
            deltas.sort_by(NumericValue::total_cmp);
            deltas.dedup_by(|left, right| approximately_equal(left.value(), right.value()));
            additive_effect_deltas.push(deltas);
        }
        let mut target_split_boundaries = task
            .numeric_variables()
            .iter()
            .enumerate()
            .filter(|(_, variable)| variable.get_type() == &NumericType::Constant)
            .filter_map(|(numeric_var_id, _)| {
                task.get_initial_numeric_state_values()
                    .get(numeric_var_id)
                    .copied()
                    .filter(|value| value.is_finite())
            })
            .map(float_tolerance::canonicalize)
            .collect::<Vec<_>>();
        target_split_boundaries.sort_by(NumericValue::total_cmp);
        target_split_boundaries.dedup_by(|left, right| left.to_bits() == right.to_bits());
        let mut prop_split_dependent_operators = (0..task.get_num_variables())
            .map(|_| OperatorBitSet::empty(operator_count))
            .collect::<Vec<_>>();
        let mut numeric_split_dependent_operators = (0..task.numeric_variables().len())
            .map(|_| OperatorBitSet::empty(operator_count))
            .collect::<Vec<_>>();
        for (op_id, op) in task.get_operators().iter().enumerate() {
            let mut prop_dependencies = vec![false; task.get_num_variables()];
            let mut numeric_dependencies = vec![false; task.numeric_variables().len()];
            let mut visiting = vec![false; task.get_num_variables()];
            for precondition in op.preconditions() {
                mark_fact_split_dependencies(
                    task,
                    precondition,
                    &propositional_axioms_by_prop_var,
                    &mut visiting,
                    &mut prop_dependencies,
                    &mut numeric_dependencies,
                )?;
            }
            for effect in op.effects() {
                let var_id = effect.var_id();
                if !task.numeric_conditions().is_condition_var(var_id)
                    && propositional_axioms_by_prop_var[var_id.index()].is_empty()
                {
                    prop_dependencies[var_id.index()] = true;
                }
            }
            for effect in op.assignment_effects() {
                let var_id = effect.affected_var_id();
                if task.numeric_variables()[var_id.index()].get_type() == &NumericType::Regular {
                    numeric_dependencies[var_id.index()] = true;
                }
            }
            for (numeric_var_id, view) in additive_numeric_views.iter().enumerate() {
                if view.as_ref().is_some_and(|view| {
                    view.operator_delta(OperatorIndex::from_usize(op_id))
                        .is_ok_and(|delta| delta.value().abs() > float_tolerance::SEARCH_EPSILON)
                }) {
                    numeric_dependencies[numeric_var_id] = true;
                }
            }
            debug_assert_eq!(
                prop_dependencies.len(),
                task.get_num_variables(),
                "operator {op_id} propositional dependency width changed"
            );
            for (var_id, depends) in prop_dependencies.into_iter().enumerate() {
                if depends {
                    prop_split_dependent_operators[var_id].insert(OperatorIndex::from_usize(op_id));
                }
            }
            for (var_id, depends) in numeric_dependencies.into_iter().enumerate() {
                if depends {
                    numeric_split_dependent_operators[var_id]
                        .insert(OperatorIndex::from_usize(op_id));
                }
            }
        }
        let icaps_random = if matches!(
            config.split_selection,
            CartesianSplitSelection::Icaps26(Icaps26SplitSelection::Random)
        ) {
            let seed = config.random_seed.unwrap_or(2011);
            ensure!(
                u32::try_from(seed).is_ok(),
                "ICAPS artifact random seed exceeds 32 bits: {seed}"
            );
            Some(RefCell::new(ArtifactMt19937::new(seed as u32)))
        } else {
            None
        };
        let random_split_rng = matches!(config.split_selection, CartesianSplitSelection::Random)
            .then(|| RefCell::new(SmallRng::seed_from_u64(config.random_seed.unwrap_or(2011))));
        let initial_numeric = task.get_initial_numeric_state_values();
        let mut numeric_integer_lattice = initial_numeric
            .iter()
            .map(|&value| approximately_equal(value.value(), value.value().round()))
            .collect::<Vec<_>>();
        for op in task.get_operators() {
            for effect in op.assignment_effects() {
                let rhs = initial_numeric[effect.var_id().index()];
                let preserves_integers = match effect.operation() {
                    AssignmentOperation::Plus
                    | AssignmentOperation::Minus
                    | AssignmentOperation::Assign
                    | AssignmentOperation::Times => {
                        approximately_equal(rhs.value(), rhs.value().round())
                    }
                    AssignmentOperation::Divide => approximately_equal(rhs.value().abs(), 1.0),
                };
                numeric_integer_lattice[effect.affected_var_id().index()] &= preserves_integers;
            }
        }
        for (numeric_var_id, view) in additive_numeric_views.iter().enumerate() {
            let Some(view) = view else {
                continue;
            };
            let initial_value = view.evaluate(initial_numeric);
            numeric_integer_lattice[numeric_var_id] =
                approximately_equal(initial_value.value(), initial_value.value().round())
                    && (0..operator_count).all(|op_id| {
                        view.operator_delta(OperatorIndex::from_usize(op_id))
                            .is_ok_and(|delta| {
                                approximately_equal(delta.value(), delta.value().round())
                            })
                    });
        }
        Ok(Self {
            task,
            propositional_axioms_by_prop_var,
            operator_costs,
            prop_split_dependent_operators,
            numeric_split_dependent_operators,
            coordinates,
            additive_effect_deltas,
            numeric_integer_lattice,
            random_seed: config.random_seed,
            random_split_rng,
            icaps_random,
            refinement_direction: config.refinement_direction,
            flaw_candidate_generation: config.flaw_candidate_generation,
            split_selection_rank: config.split_selection_rank,
            split_selection: config.split_selection,
            target_split_boundaries,
        })
    }

    pub(super) fn choose_keyed_index(&self, keys: &[u64], tag: u64) -> usize {
        debug_assert!(
            !keys.is_empty(),
            "cannot choose from an empty Cartesian candidate set"
        );
        let Some(seed) = self.random_seed else {
            return 0;
        };
        keys.iter()
            .enumerate()
            .min_by_key(|(_, key)| mix_seed(seed ^ tag ^ **key))
            .map(|(index, _)| index)
            .expect("nonempty Cartesian key set has no minimum")
    }

    pub(super) fn choose_split_index(&self, candidates: &[Split], tag: u64) -> usize {
        debug_assert!(
            !candidates.is_empty(),
            "cannot choose from an empty split set"
        );
        if let Some(rank) = self.split_selection_rank {
            let mut indices = (0..candidates.len()).collect::<Vec<_>>();
            indices.sort_by_key(|&index| {
                let dimension = match candidates[index].dimension() {
                    SplitDimension::Propositional(var_id) => (0usize, var_id),
                    SplitDimension::Numeric(var_id) => (1usize, var_id),
                };
                (dimension, split_choice_key(self, &candidates[index]))
            });
            return indices[rank % indices.len()];
        }
        let keys = candidates
            .iter()
            .map(|split| split_choice_key(self, split))
            .collect::<Vec<_>>();
        self.choose_keyed_index(&keys, tag)
    }

    pub(super) fn choose_icaps_random_index(&self, candidate_count: usize) -> usize {
        debug_assert!(candidate_count > 0, "cannot choose from an empty split set");
        self.icaps_random
            .as_ref()
            .expect("ICAPS random selector must initialize its RNG")
            .borrow_mut()
            .uniform_index(candidate_count)
    }

    pub(super) fn choose_random_split_index(&self, candidate_count: usize) -> usize {
        debug_assert!(candidate_count > 0, "cannot choose from an empty split set");
        self.random_split_rng
            .as_ref()
            .expect("native random selector must initialize its RNG")
            .borrow_mut()
            .gen_range(0..candidate_count)
    }

    pub(super) fn operator_depends_on_split(
        &self,
        op_id: OperatorIndex,
        dimension: SplitDimension,
    ) -> bool {
        self.split_dependent_operators(dimension).contains(op_id)
    }

    pub(super) fn split_dependent_operators(&self, dimension: SplitDimension) -> &OperatorBitSet {
        match dimension {
            SplitDimension::Propositional(var_id) => {
                &self.prop_split_dependent_operators[var_id.index()]
            }
            SplitDimension::Numeric(var_id) => {
                &self.numeric_split_dependent_operators[var_id.index()]
            }
        }
    }

    pub(super) fn invariant_split_dimension_overlaps(
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

    pub(super) fn may_transition_after_independent_split(
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

    pub(super) fn trivial_region(&self) -> Result<CartesianSet> {
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
                    Interval::singleton(float_tolerance::canonicalize(initial_numeric[var_id]))
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

    pub(super) fn region_admits_fact(
        &self,
        region: &CartesianSet,
        fact: &ExplicitFact,
    ) -> Result<bool> {
        let mut visiting = vec![false; self.task.get_num_variables()];
        self.region_admits_fact_inner(region, fact, &mut visiting)
    }

    pub(super) fn region_admits_fact_inner(
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

    pub(super) fn region_guarantees_fact(
        &self,
        region: &CartesianSet,
        fact: &ExplicitFact,
    ) -> Result<bool> {
        let mut visiting = vec![false; self.task.get_num_variables()];
        self.region_guarantees_fact_inner(region, fact, &mut visiting)
    }

    pub(super) fn region_guarantees_fact_inner(
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

    pub(super) fn all_conditions_admitted(
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

    pub(super) fn all_conditions_guaranteed(
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

    pub(super) fn propositional_axiom_default(&self, var_id: usize) -> Result<ExplicitValueIndex> {
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

    pub(super) fn comparison_truths(
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

    pub(super) fn operator_may_apply(
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

    pub(super) fn propositional_dimension_may_transition(
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
        if matches!(self.split_selection, CartesianSplitSelection::Icaps26(_))
            && let Some(precondition) = op
                .preconditions()
                .iter()
                .find(|precondition| precondition.var_index() == var_id)
        {
            return target.propositions()[var_id.index()]
                .binary_search(&(precondition.value_index()))
                .is_ok();
        }
        sorted_value_sets_overlap(
            &source.propositions()[var_id.index()],
            &target.propositions()[var_id.index()],
        )
    }

    pub(super) fn split_dimension_may_transition(
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

    pub(super) fn numeric_dimension_may_transition(
        &self,
        source: Interval,
        op_id: OperatorIndex,
        target: Interval,
        var_id: VariableIndex,
    ) -> Result<bool> {
        let Some(preimage) = self.numeric_effect_preimage(target, op_id, var_id)? else {
            return Ok(false);
        };
        let source = if matches!(self.split_selection, CartesianSplitSelection::Icaps26(_)) {
            source.intersection(&self.icaps_numeric_precondition(op_id, var_id)?)
        } else {
            source
        };
        Ok(!source.is_empty() && preimage.intersects(&source))
    }

    pub(super) fn icaps_numeric_precondition(
        &self,
        op_id: OperatorIndex,
        var_id: VariableIndex,
    ) -> Result<Interval> {
        let mut interval = Interval::unbounded();
        for fact in self.task.get_operators()[op_id.index()].preconditions() {
            let Some(tree_id) = self.task.numeric_conditions().id_for_var(fact.var_index()) else {
                continue;
            };
            let (condition_var_id, condition) =
                desired_comparison_interval(self, tree_id, fact.value_index())?;
            if condition_var_id == var_id {
                interval = interval.intersection(&condition);
            }
        }
        Ok(interval)
    }

    pub(super) fn parent_loop_source_to_split_children(
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

    pub(super) fn may_transition(
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

    pub(super) fn numeric_effect_preimage(
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

    pub(super) fn source_set_source_for_transition(
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

    pub(super) fn region_is_goal(&self, region: &CartesianSet) -> Result<bool> {
        for goal_id in 0..self.task.get_num_goals() {
            if !self.region_admits_fact(region, self.task.get_goal_fact(goal_id))? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    pub(super) fn concrete_prop_values(
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
