use planforge_sas::numeric_task::{ExplicitValueIndex, VariableIndex};

use super::*;

#[derive(Debug, Clone, Default)]
struct MatchTreeNode {
    value_children: Vec<Option<Box<MatchTreeNode>>>,
    wildcard_child: Option<Box<MatchTreeNode>>,
    ops: Vec<usize>,
}

/// The abstract operators of an abstraction indexed by the abstract states
/// they regress from: a decision tree over the variables their regression
/// preconditions mention, most frequent first.
#[derive(Debug, Clone)]
pub(crate) struct MatchTree {
    layout: Arc<StateLayout>,
    var_order: Vec<usize>,
    root: MatchTreeNode,
}

fn fact_value_for_var(facts: &[ExplicitFact], var: usize) -> Option<usize> {
    facts
        .binary_search_by_key(&var, |fact| fact.var())
        .ok()
        .map(|index| facts[index].value())
}

impl MatchTree {
    /// Indexes `operators` of the abstraction with `layout` by the abstract
    /// states they regress from.
    pub(crate) fn new(layout: &Arc<StateLayout>, operators: &[AbstractOperator]) -> Self {
        let mut var_counts = vec![0usize; layout.multipliers().len()];
        for op in operators {
            for fact in &op.regression_preconditions {
                if layout.radix(fact.var()) > 1 {
                    var_counts[fact.var()] += 1;
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
            layout: Arc::clone(layout),
            var_order,
            root: MatchTreeNode::default(),
        };
        for (op_id, op) in operators.iter().enumerate() {
            tree.insert(op_id, &op.regression_preconditions);
        }
        tree
    }

    fn insert(&mut self, op_id: usize, conds: &[ExplicitFact]) {
        let mut node = &mut self.root;
        for &var in &self.var_order {
            node = match fact_value_for_var(conds, var) {
                Some(value) => {
                    let radix = self.layout.radix(var);
                    if node.value_children.len() < radix {
                        node.value_children.resize_with(radix, || None);
                    }
                    node.value_children[value].get_or_insert_with(Box::default)
                }
                None => node.wildcard_child.get_or_insert_with(Box::default),
            };
        }
        node.ops.push(op_id);
    }

    pub(super) fn get_applicable_operator_ids(&self, state: usize, out: &mut Vec<usize>) {
        out.clear();
        self.collect_applicable(&self.root, 0, state, out);
    }

    fn collect_applicable(
        &self,
        node: &MatchTreeNode,
        depth: usize,
        state: usize,
        out: &mut Vec<usize>,
    ) {
        if depth == self.var_order.len() {
            out.extend_from_slice(&node.ops);
            return;
        }
        let value = self.layout.value(state, self.var_order[depth]);
        if let Some(child) = node.value_children.get(value).and_then(Option::as_deref) {
            self.collect_applicable(child, depth + 1, state, out);
        }
        if let Some(child) = node.wildcard_child.as_deref() {
            self.collect_applicable(child, depth + 1, state, out);
        }
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

    /// The region of the concrete states in abstract state `state`.
    pub(crate) fn state_region(&self, state: usize) -> CartesianSet {
        let propositions = (0..self.layout.domain_sizes().len())
            .map(|var_id| {
                self.concrete_values_for_abstract_value(
                    VariableIndex::from_usize(var_id),
                    ExplicitValueIndex::from_usize(self.layout.value(state, var_id)),
                )
            })
            .collect();
        let numeric = (0..self.layout.numeric_domain_sizes().len())
            .map(VariableIndex::from_usize)
            .map(|numeric_var_id| {
                self.partitions.partition_interval(
                    numeric_var_id,
                    self.layout.partition(state, numeric_var_id),
                )
            })
            .collect();
        let region = CartesianSet::with_constrained_props(
            propositions,
            numeric,
            Arc::clone(&self.state_constrained_props),
        );
        #[cfg(debug_assertions)]
        self.debug_assert_constrained_props_cover_narrowings(&region);
        region
    }

    /// The region of the concrete states that satisfy the abstract `facts`.
    pub(super) fn cartesian_set_from_facts(
        &self,
        task: &dyn AbstractNumericTask,
        facts: &[ExplicitFact],
    ) -> CartesianSet {
        let num_props = self.layout.domain_sizes().len();
        let mut propositions = self.full_propositional_region();
        let mut numeric = vec![Interval::unbounded(); task.numeric_variables().len()];
        // Only the dimensions named by a fact are narrowed below; the rest keep
        // their whole domain. Recording them here is exact rather than
        // rediscovered by comparing lengths, except when a variable has a single
        // abstract class covering its domain -- then this is a sound superset.
        let mut constrained_props = Vec::new();

        for fact in facts {
            if fact.var() < num_props {
                propositions[fact.var()] =
                    self.concrete_values_for_abstract_value(fact.var_index(), fact.value_index());
                constrained_props
                    .push(u32::try_from(fact.var()).expect("propositional var id exceeds u32"));
            } else {
                let numeric_var_id = VariableIndex::from_usize(fact.var() - num_props);
                numeric[numeric_var_id.index()] = self
                    .partitions
                    .partition_interval(numeric_var_id, fact.value());
            }
        }

        constrained_props.sort_unstable();
        constrained_props.dedup();
        let region =
            CartesianSet::with_constrained_props(propositions, numeric, constrained_props.into());
        #[cfg(debug_assertions)]
        self.debug_assert_constrained_props_cover_narrowings(&region);
        region
    }

    pub(super) fn full_propositional_region(&self) -> Vec<Vec<ExplicitValueIndex>> {
        self.domain_mapping
            .iter()
            .map(|mapping| {
                (0..mapping.len())
                    .map(ExplicitValueIndex::from_usize)
                    .collect()
            })
            .collect()
    }

    /// The concrete values `abstract_value` of `var_id` stands for; never
    /// empty, since the domain mapping is onto its abstract values.
    pub(super) fn concrete_values_for_abstract_value(
        &self,
        var_id: VariableIndex,
        abstract_value: ExplicitValueIndex,
    ) -> Vec<ExplicitValueIndex> {
        // `filter_map().collect()` preallocates capacity matching the inner iterator's
        // upper-bound size_hint (here the full domain mapping length), so a var that
        // only has a handful of concrete values mapped to this abstract slot leaves
        // most of that capacity unused. Shrinking before returning saves typically
        // 50-90% of the per-inner-`Vec` heap allocation, which on SCP runs over many
        // state regions dominates the propositional representation.
        let mut values = self.domain_mapping[var_id.index()]
            .iter()
            .enumerate()
            .filter_map(|(concrete_value, &mapped_value)| {
                (mapped_value == abstract_value)
                    .then_some(ExplicitValueIndex::from_usize(concrete_value))
            })
            .collect::<Vec<ExplicitValueIndex>>();
        debug_assert!(
            !values.is_empty(),
            "abstract value {abstract_value} of variable {var_id} has no concrete value"
        );
        values.shrink_to_fit();
        values
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
        for fact in task.goals() {
            let var = fact.var_index();
            if self.numeric_conditions.is_condition_var(var) {
                out.push(*fact);
                continue;
            }
            if self.layout.domain_sizes()[var.index()] <= 1 {
                continue;
            }
            let mapped = self.domain_mapping[var.index()][fact.value()];
            out.push(ExplicitFact::propositional(var.index(), mapped.index()));
        }

        out
    }

    /// Whether abstract state `state` satisfies the abstract `goals`: its
    /// propositional values match, and its numeric partitions admit every
    /// goal condition. `intervals` is scratch space.
    pub fn is_goal_state(
        &self,
        task: &dyn AbstractNumericTask,
        state: usize,
        goals: &[ExplicitFact],
        intervals: &mut Vec<Interval>,
    ) -> Result<bool> {
        let mut intervals_filled = false;
        for goal in goals {
            if let Some(condition) = self.numeric_conditions.for_var(goal.var_index()) {
                if !intervals_filled {
                    fill_state_intervals(
                        task,
                        self.numeric_conditions.all(),
                        &self.partitions,
                        &self.layout,
                        state,
                        intervals,
                    )?;
                    intervals_filled = true;
                }
                let contradicting = Some(goal.value() != ConditionValue::True.as_usize());
                if condition.evaluate_interval(intervals) == contradicting {
                    return Ok(false);
                }
            } else if self.layout.value(state, goal.var()) != goal.value() {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// The abstract states that satisfy the abstract `goals`, ascending.
    pub(crate) fn goal_states(
        &self,
        task: &dyn AbstractNumericTask,
        goals: &[ExplicitFact],
        deadline: Deadline,
    ) -> Result<Vec<usize>> {
        let mut intervals = Vec::new();
        let mut goal_states = Vec::new();
        for state in 0..self.layout.num_states() {
            if state.is_multiple_of(1024) {
                deadline.ensure(DISTANCE_TABLE)?;
            }
            if self.is_goal_state(task, state, goals, &mut intervals)? {
                goal_states.push(state);
            }
        }
        Ok(goal_states)
    }

    /// The abstract state of the task's initial state.
    pub(crate) fn initial_state(&self, task: &dyn AbstractNumericTask) -> Result<usize> {
        self.abstract_state(
            task.get_initial_propositional_state_values(),
            task.get_initial_numeric_state_values(),
        )
    }

    /// The abstract state containing the concrete state with these values.
    /// Coordinates with an additive view are evaluated through it.
    pub(crate) fn abstract_state(
        &self,
        propositions: &[ExplicitValueIndex],
        numeric: &[NumericValue],
    ) -> Result<usize> {
        let domain_sizes = self.layout.domain_sizes();
        let numeric_domain_sizes = self.layout.numeric_domain_sizes();
        ensure!(
            propositions.len() == domain_sizes.len(),
            "concrete state has {} propositional values, the abstraction {}",
            propositions.len(),
            domain_sizes.len()
        );
        ensure!(
            numeric.len() == numeric_domain_sizes.len(),
            "concrete state has {} numeric values, the abstraction {}",
            numeric.len(),
            numeric_domain_sizes.len()
        );
        let multipliers = self.layout.multipliers();

        // A variable with one abstract value contributes nothing.
        let mut state: usize = 0;
        for (var, &concrete_value) in propositions.iter().enumerate() {
            if domain_sizes[var] == 1 {
                continue;
            }
            let abstract_value = *self.domain_mapping[var]
                .get(concrete_value.index())
                .with_context(|| {
                    format!("propositional var {var} has no value {concrete_value}")
                })?;
            state += multipliers[var] * abstract_value.index();
        }

        for num_var_id in (0..numeric_domain_sizes.len()).map(VariableIndex::from_usize) {
            if numeric_domain_sizes[num_var_id.index()] == 1 {
                continue;
            }
            let abs_var = abstraction_numeric_var(domain_sizes.len(), num_var_id);
            let concrete_value = match self.additive_numeric_views.get(num_var_id.index()) {
                Some(view) => view.evaluate(numeric),
                None => numeric[num_var_id.index()],
            };
            let val = float_tolerance::canonicalize(concrete_value);
            ensure!(
                val.is_finite(),
                "numeric value for var {num_var_id} must be finite, got {val}"
            );
            let part = partition_for_value(self.partitions.partitions(num_var_id), val)
                .with_context(|| {
                    format!("numeric value {val} is in no partition of numeric var {num_var_id}")
                })?;
            state += multipliers[abs_var.index()] * part;
        }

        Ok(state)
    }
}
