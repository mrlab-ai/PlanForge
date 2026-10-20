use planforge_sas::numeric_task::{ExplicitValueIndex, VariableIndex};

use super::*;

#[derive(Debug, Clone, Default)]
struct MatchTreeNode {
    value_children: Vec<Option<Box<MatchTreeNode>>>,
    wildcard_child: Option<Box<MatchTreeNode>>,
    ops: Vec<usize>,
}

#[derive(Debug, Clone)]
pub(super) struct MatchTree {
    var_order: Vec<usize>,
    domain_sizes: Vec<usize>,
    numeric_domain_sizes: Vec<usize>,
    hash_multipliers: Vec<usize>,
    root: MatchTreeNode,
}

fn domain_size_for_var(
    domain_sizes: &[usize],
    numeric_domain_sizes: &[usize],
    var: usize,
) -> usize {
    if var < domain_sizes.len() {
        domain_sizes.get(var).copied().unwrap_or(0)
    } else {
        numeric_domain_sizes
            .get(var - domain_sizes.len())
            .copied()
            .unwrap_or(0)
    }
}

fn fact_value_for_var(facts: &[ExplicitFact], var: usize) -> Option<usize> {
    facts
        .binary_search_by_key(&var, |fact| fact.var())
        .ok()
        .map(|index| facts[index].value())
}

impl MatchTree {
    /// Indexes `operators`, built by `generator`, by the abstract states they
    /// regress from.
    pub(super) fn new(
        generator: &AbstractOperatorGenerator,
        operators: &[AbstractOperator],
    ) -> Self {
        let domain_sizes = generator.domain_sizes();
        let numeric_domain_sizes = generator.numeric_domain_sizes();
        let hash_multipliers = generator.hash_multipliers();
        let total_vars = domain_sizes.len() + numeric_domain_sizes.len();
        let mut var_counts = vec![0usize; total_vars];
        for op in operators {
            for f in op.regression_preconditions.iter() {
                if f.var() >= total_vars {
                    continue;
                }
                let domain_size = domain_size_for_var(domain_sizes, numeric_domain_sizes, f.var());
                if domain_size > 1 {
                    var_counts[f.var()] += 1;
                }
            }
        }
        let mut var_order: Vec<usize> = var_counts
            .iter()
            .enumerate()
            .filter_map(|(var, &count)| (count > 0).then_some(var))
            .collect();
        var_order.sort_by(|&left, &right| {
            var_counts[right]
                .cmp(&var_counts[left])
                .then_with(|| left.cmp(&right))
        });

        let mut tree = Self {
            var_order,
            domain_sizes: domain_sizes.to_vec(),
            numeric_domain_sizes: numeric_domain_sizes.to_vec(),
            hash_multipliers: hash_multipliers.to_vec(),
            root: MatchTreeNode::default(),
        };

        for (op_id, op) in operators.iter().enumerate() {
            tree.insert(op_id, &op.regression_preconditions);
        }

        tree
    }

    fn insert(&mut self, op_id: usize, conds: &[ExplicitFact]) {
        fn insert_rec(
            node: &mut MatchTreeNode,
            depth: usize,
            var_order: &[usize],
            conds: &[ExplicitFact],
            domain_sizes: &[usize],
            numeric_domain_sizes: &[usize],
            op_id: usize,
        ) {
            if depth == var_order.len() {
                node.ops.push(op_id);
                return;
            }
            let var = var_order[depth];
            if let Some(val) = fact_value_for_var(conds, var) {
                let domain_size = domain_size_for_var(domain_sizes, numeric_domain_sizes, var);
                if node.value_children.len() < domain_size {
                    node.value_children.resize_with(domain_size, || None);
                }
                let child = node.value_children[val]
                    .get_or_insert_with(|| Box::new(MatchTreeNode::default()));
                insert_rec(
                    child.as_mut(),
                    depth + 1,
                    var_order,
                    conds,
                    domain_sizes,
                    numeric_domain_sizes,
                    op_id,
                );
            } else {
                let child = node
                    .wildcard_child
                    .get_or_insert_with(|| Box::new(MatchTreeNode::default()));
                insert_rec(
                    child.as_mut(),
                    depth + 1,
                    var_order,
                    conds,
                    domain_sizes,
                    numeric_domain_sizes,
                    op_id,
                );
            }
        }

        insert_rec(
            &mut self.root,
            0,
            &self.var_order,
            conds,
            &self.domain_sizes,
            &self.numeric_domain_sizes,
            op_id,
        );
    }

    pub(super) fn get_applicable_operator_ids(&self, state_hash: usize, out: &mut Vec<usize>) {
        out.clear();
        self.collect_applicable(&self.root, 0, state_hash, out);
    }

    fn collect_applicable(
        &self,
        node: &MatchTreeNode,
        depth: usize,
        state_hash: usize,
        out: &mut Vec<usize>,
    ) {
        if depth == self.var_order.len() {
            out.extend_from_slice(&node.ops);
            return;
        }
        let var = self.var_order[depth];
        let actual = self.get_var_value(state_hash, var);
        if let Some(child) = node.value_children.get(actual).and_then(Option::as_deref) {
            self.collect_applicable(child, depth + 1, state_hash, out);
        }
        if let Some(child) = node.wildcard_child.as_deref() {
            self.collect_applicable(child, depth + 1, state_hash, out);
        }
    }
    fn get_var_value(&self, state_hash: usize, var: usize) -> usize {
        let num_props = self.domain_sizes.len();
        debug_assert!(
            var < self.hash_multipliers.len(),
            "match tree var out of bounds for hash multipliers: {} >= {}",
            var,
            self.hash_multipliers.len()
        );
        let Some(mult) = self.hash_multipliers.get(var).copied() else {
            return 0;
        };
        let state = state_hash;
        let dom_size = if var < num_props {
            debug_assert!(
                var < self.domain_sizes.len(),
                "match tree propositional var out of bounds: {} >= {}",
                var,
                self.domain_sizes.len()
            );
            self.domain_sizes.get(var).copied().unwrap_or(0)
        } else {
            let n = var - num_props;
            debug_assert!(
                n < self.numeric_domain_sizes.len(),
                "match tree numeric var out of bounds: {} >= {}",
                n,
                self.numeric_domain_sizes.len()
            );
            self.numeric_domain_sizes.get(n).copied().unwrap_or(0)
        };
        debug_assert!(
            dom_size > 0,
            "match tree domain size must be positive for var {var}"
        );
        if dom_size == 0 {
            return 0;
        }

        (state / mult) % dom_size
    }
}

impl DomainAbstractionFactory {
    /// Check the invariant the sparse propositional overlap depends on: a
    /// dimension the region does not list as constrained must admit its whole
    /// concrete domain. Violating it makes disjoint regions compare as
    /// overlapping, so an operator's cost can be claimed twice -- a wrong
    /// heuristic rather than a slow one.
    #[cfg(debug_assertions)]
    fn debug_assert_constrained_props_cover_narrowings(&self, region: &CartesianSet) {
        for (var_id, values) in region.propositions().iter().enumerate() {
            let concrete_size = self.domain_mapping[var_id].len();
            let listed = region
                .constrained_props()
                .binary_search(&(var_id as u32))
                .is_ok();
            assert!(
                listed || values.len() == concrete_size,
                "state region narrows propositional var {var_id} to {} of {concrete_size} values \
                 without listing it as constrained (listed dimensions: {:?})",
                values.len(),
                region.constrained_props()
            );
        }
    }

    pub(crate) fn cartesian_set_from_hash(
        &self,
        state_hash: usize,
        numeric_domain_sizes: &[usize],
        hash_multipliers: &[usize],
    ) -> Result<CartesianSet> {
        let region = CartesianSet::with_constrained_props(
            self.propositional_region_from_hash(state_hash, hash_multipliers)?,
            self.numeric_region_from_hash(state_hash, numeric_domain_sizes, hash_multipliers)?,
            Arc::clone(&self.state_constrained_props),
        );
        #[cfg(debug_assertions)]
        self.debug_assert_constrained_props_cover_narrowings(&region);
        Ok(region)
    }

    pub(super) fn cartesian_set_from_facts(
        &self,
        task: &dyn AbstractNumericTask,
        facts: &[ExplicitFact],
    ) -> Result<CartesianSet> {
        let num_props = self.domain_sizes.len();
        let mut propositions = self.full_propositional_region()?;
        let mut numeric = vec![Interval::unbounded(); task.numeric_variables().len()];
        // Only the dimensions named by a fact are narrowed below; the rest keep
        // their whole domain. Recording them here is exact rather than
        // rediscovered by comparing lengths, except when a variable has a single
        // abstract class covering its domain -- then this is a sound superset.
        let mut constrained_props = Vec::new();

        for fact in facts {
            if fact.var() < num_props {
                propositions[fact.var()] =
                    self.concrete_values_for_abstract_value(fact.var_index(), fact.value_index())?;
                constrained_props
                    .push(u32::try_from(fact.var()).expect("propositional var id exceeds u32"));
            } else {
                let numeric_var_id = fact.var() - num_props;
                ensure!(
                    numeric_var_id < numeric.len(),
                    "abstract-operator region fact references numeric var {numeric_var_id}, but task has {} numeric vars",
                    numeric.len()
                );
                numeric[numeric_var_id] = self
                    .partitions
                    .partition_interval(VariableIndex::from_usize(numeric_var_id), fact.value())
                    .with_context(|| {
                        format!(
                            "missing interval for numeric var {numeric_var_id} partition {}",
                            fact.value()
                        )
                    })?;
            }
        }

        constrained_props.sort_unstable();
        constrained_props.dedup();
        let region =
            CartesianSet::with_constrained_props(propositions, numeric, constrained_props.into());
        #[cfg(debug_assertions)]
        self.debug_assert_constrained_props_cover_narrowings(&region);
        Ok(region)
    }

    pub(super) fn full_propositional_region(&self) -> Result<Vec<Vec<ExplicitValueIndex>>> {
        let mut region = Vec::with_capacity(self.domain_sizes.len());
        for var_id in 0..self.domain_sizes.len() {
            let mapping = self
                .domain_mapping
                .get(var_id)
                .with_context(|| format!("missing domain mapping for var {var_id}"))?;
            ensure!(
                !mapping.is_empty(),
                "empty concrete value set for propositional var {var_id}"
            );
            region.push(
                (0..mapping.len())
                    .map(ExplicitValueIndex::from_usize)
                    .collect(),
            );
        }
        Ok(region)
    }

    pub(super) fn concrete_values_for_abstract_value(
        &self,
        var_id: VariableIndex,
        abstract_value: ExplicitValueIndex,
    ) -> Result<Vec<ExplicitValueIndex>> {
        // `filter_map().collect()` preallocates capacity matching the inner iterator's
        // upper-bound size_hint (here the full domain mapping length), so a var that
        // only has a handful of concrete values mapped to this abstract slot leaves
        // most of that capacity unused. Shrinking before returning saves typically
        // 50-90% of the per-inner-`Vec` heap allocation, which on SCP runs over many
        // state regions dominates the propositional representation.
        let mut values = self
            .domain_mapping
            .get(var_id.index())
            .with_context(|| format!("missing domain mapping for var {var_id}"))?
            .iter()
            .enumerate()
            .filter_map(|(concrete_value, &mapped_value)| {
                (mapped_value == abstract_value)
                    .then_some(ExplicitValueIndex::from_usize(concrete_value))
            })
            .collect::<Vec<ExplicitValueIndex>>();
        ensure!(
            !values.is_empty(),
            "empty concrete value set for var {var_id} abstract value {abstract_value}"
        );
        values.shrink_to_fit();
        Ok(values)
    }

    pub(super) fn propositional_region_from_hash(
        &self,
        state_hash: usize,
        hash_multipliers: &[usize],
    ) -> Result<Vec<Vec<ExplicitValueIndex>>> {
        let mut region = Vec::with_capacity(self.domain_sizes.len());
        for (var_id, &domain_size) in self.domain_sizes.iter().enumerate() {
            ensure!(domain_size > 0, "domain size must be > 0 for var {var_id}");
            let multiplier = *hash_multipliers
                .get(var_id)
                .with_context(|| format!("missing hash multiplier for var {var_id}"))?;
            let abstract_value = (state_hash / multiplier) % domain_size;
            region.push(self.concrete_values_for_abstract_value(
                VariableIndex::from_usize(var_id),
                ExplicitValueIndex::from_usize(abstract_value),
            )?);
        }
        Ok(region)
    }

    pub(super) fn numeric_region_from_hash(
        &self,
        state_hash: usize,
        numeric_domain_sizes: &[usize],
        hash_multipliers: &[usize],
    ) -> Result<Vec<Interval>> {
        let num_props = self.domain_sizes.len();
        let mut region = Vec::with_capacity(numeric_domain_sizes.len());
        for (numeric_var_id, &domain_size) in numeric_domain_sizes.iter().enumerate() {
            let numeric_var_id = VariableIndex::from_usize(numeric_var_id);
            ensure!(
                domain_size > 0,
                "numeric domain size must be > 0 for var {numeric_var_id}"
            );
            let abs_var_id = abstraction_numeric_var(num_props, numeric_var_id);
            let multiplier = *hash_multipliers.get(abs_var_id.index()).with_context(|| {
                format!("missing hash multiplier for numeric var {numeric_var_id}")
            })?;
            let partition_id = (state_hash / multiplier) % domain_size;
            let interval = self
                .partitions
                .partition_interval(numeric_var_id, partition_id)
                .with_context(|| {
                    format!(
                        "missing interval for numeric var {numeric_var_id} partition {partition_id}"
                    )
                })?;
            region.push(interval);
        }
        Ok(region)
    }

    /// The task's goals in abstract values. Goals on condition variables keep
    /// their concrete value: conditions are derived from the numeric
    /// partitions, and [`Self::is_goal_state`] evaluates them there.
    ///
    /// Every goal fact is one the abstraction can reach --
    /// `validate_abstractable_goal` has already refused a goal on a derived
    /// variable, which no abstract operator writes. A variable the abstraction has
    /// collapsed to a single value carries no information and is dropped rather
    /// than mapped.
    pub(super) fn compute_abstract_goals(
        &self,
        task: &dyn AbstractNumericTask,
    ) -> Vec<ExplicitFact> {
        let mut out: Vec<ExplicitFact> = Vec::new();
        for goal_index in 0..task.get_num_goals() {
            let fact = task.get_goal_fact(goal_index);
            let var = fact.var_index();
            if self.numeric_conditions.is_condition_var(var) {
                out.push(*fact);
                continue;
            }
            if self.domain_sizes[var.index()] <= 1 {
                continue;
            }
            let mapped = self.domain_mapping[var.index()][fact.value()];
            out.push(ExplicitFact::propositional(var.index(), mapped.index()));
        }

        out
    }

    /// Whether abstract state `state_hash` satisfies the abstract `goals`:
    /// its propositional values match, and its numeric partitions admit every
    /// goal condition.
    pub fn is_goal_state(
        &self,
        task: &dyn AbstractNumericTask,
        state_hash: usize,
        goals: &[ExplicitFact],
        numeric_domain_sizes: &[usize],
        hash_multipliers: &[usize],
    ) -> Result<bool> {
        let num_props = self.domain_sizes.len();
        let mut numeric_inputs: Option<Vec<Interval>> = None;
        for goal in goals {
            let var = goal.var();
            if let Some(condition) = self.numeric_conditions.for_var(goal.var_index()) {
                let inputs = match &numeric_inputs {
                    Some(inputs) => inputs,
                    None => {
                        numeric_inputs.insert(prepare_comparison_tree_inputs_from_abstract_state(
                            task,
                            self.numeric_conditions.all(),
                            &self.partitions,
                            AbstractStateHash {
                                hash: state_hash,
                                num_props,
                                numeric_domain_sizes,
                                hash_multipliers,
                            },
                        )?)
                    }
                };
                let contradicting = Some(goal.value() != ConditionValue::True.as_usize());
                if condition.evaluate_interval(inputs) == contradicting {
                    return Ok(false);
                }
                continue;
            }
            let dom_size = if var < num_props {
                self.domain_sizes[var]
            } else {
                numeric_domain_sizes[var - num_props]
            };
            if (state_hash / hash_multipliers[var]) % dom_size != goal.value() {
                return Ok(false);
            }
        }
        Ok(true)
    }

    pub(super) fn compute_initial_state_hash_determined(
        &self,
        task: &dyn AbstractNumericTask,
        numeric_domain_sizes: &[usize],
        hash_multipliers: &[usize],
    ) -> Result<usize> {
        self.abstract_state_hash(
            task.get_initial_propositional_state_values(),
            task.get_initial_numeric_state_values(),
            numeric_domain_sizes,
            hash_multipliers,
        )
    }

    /// The hash of the abstract state containing the concrete state with
    /// these values. Coordinates with an additive view are evaluated through
    /// it.
    pub(crate) fn abstract_state_hash(
        &self,
        propositions: &[ExplicitValueIndex],
        numeric: &[NumericValue],
        numeric_domain_sizes: &[usize],
        hash_multipliers: &[usize],
    ) -> Result<usize> {
        let num_props = self.domain_sizes.len();
        ensure!(
            propositions.len() == num_props,
            "concrete state has {} propositional values, the abstraction {num_props}",
            propositions.len()
        );
        ensure!(
            numeric.len() == numeric_domain_sizes.len(),
            "concrete state has {} numeric values, the abstraction {}",
            numeric.len(),
            numeric_domain_sizes.len()
        );

        // A variable with one abstract value contributes nothing.
        let mut index: usize = 0;
        for (var, &concrete_value) in propositions.iter().enumerate() {
            if self.domain_sizes[var] == 1 {
                continue;
            }
            let abstract_value = *self.domain_mapping[var]
                .get(concrete_value.index())
                .with_context(|| {
                    format!(
                        "missing mapping for propositional var {var} value index {concrete_value}"
                    )
                })?;
            index += hash_multipliers[var] * abstract_value.index();
        }

        for num_var_id in (0..numeric_domain_sizes.len()).map(VariableIndex::from_usize) {
            if numeric_domain_sizes[num_var_id.index()] == 1 {
                continue;
            }
            let abs_var = abstraction_numeric_var(num_props, num_var_id);
            let concrete_value = self
                .additive_numeric_views
                .get(num_var_id.index())
                .map(|view| view.evaluate(numeric))
                .unwrap_or(numeric[num_var_id.index()]);
            let val = float_tolerance::canonicalize(concrete_value);
            ensure!(
                val.is_finite(),
                "numeric value for var {num_var_id} must be finite, got {val}"
            );
            let parts = self
                .partitions
                .partitions(num_var_id)
                .with_context(|| format!("missing partitions for numeric var {num_var_id}"))?;
            let part = partition_for_value(parts, val).with_context(|| {
                format!("numeric value {val} is in no partition of numeric var {num_var_id}")
            })?;
            index += hash_multipliers[abs_var.index()] * part;
        }

        Ok(index)
    }
}
