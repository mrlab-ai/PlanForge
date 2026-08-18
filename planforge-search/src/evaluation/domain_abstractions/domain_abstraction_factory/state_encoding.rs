use super::*;

/// What one comparison-enumeration loop remembers across calls: the resolved
/// successor sets, how many states they hold, and the buffer that serves a
/// result too large to keep.
#[derive(Default)]
pub(super) struct ComparisonEnumerationMemo {
    pub(super) cache: ComparisonEnumerationCache,
    pub(super) cached_state_count: usize,
    pub(super) overflow: Vec<usize>,
}

const COMPARISON_ENUMERATION_CACHE_MAX_ENTRIES: usize = 2_000_000;
const COMPARISON_ENUMERATION_CACHE_MAX_STATES: usize = 10_000_000;

/// Cache used by `enumerate_states_with_evaluated_comparisons_cached`. Keyed
/// by a precomputed 64-bit signature of `(base_state_hash, fixed_comparisons)`
/// — `comparison_var_ids` is intentionally omitted because every call site
/// in this factory passes the same slice (`self.comparison_var_ids()`), so it
/// doesn't disambiguate.
///
/// The previous design hashed a `(usize, Vec<usize>, Vec<(usize, usize)>)`
/// struct via SipHash on every cache lookup; on a 200k-state minecraft build
/// `sip::Hasher::write` reached 11% of total CPU and the per-lookup
/// `Vec::to_vec`/`collect` allocations dominated `_int_malloc`. The new
/// table is a `HashMap<u64, Vec<usize>>` with an identity hasher: lookup is
/// a single load + probe, no allocation, no hash function.
type ComparisonEnumerationCache = HashMap<
    u64,
    Vec<usize>,
    std::hash::BuildHasherDefault<planforge_sas::utils::hashing::IdentityU64Hasher>,
>;

#[inline]
fn comparison_enumeration_signature(
    base_state_hash: usize,
    fixed_comparisons: &[ExplicitFact],
) -> u64 {
    // FNV-1a u64 mix + a SplitMix64 finalizer for even bit distribution
    // (same construction as `compute_signature_hash` for abstract operators).
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x00000100000001B3;

    let mut h = FNV_OFFSET;
    h ^= base_state_hash as u64;
    h = h.wrapping_mul(FNV_PRIME);
    h ^= fixed_comparisons.len() as u64;
    h = h.wrapping_mul(FNV_PRIME);
    for fact in fixed_comparisons {
        h ^= fact.var() as u64;
        h = h.wrapping_mul(FNV_PRIME);
        h ^= fact.value() as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }

    h ^= h >> 33;
    h = h.wrapping_mul(0xff51afd7ed558ccd);
    h ^= h >> 33;
    h = h.wrapping_mul(0xc4ceb9fe1a85ec53);
    h ^= h >> 33;
    h
}

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
    pub(super) fn build(
        domain_sizes: &[usize],
        numeric_domain_sizes: &[usize],
        hash_multipliers: &[usize],
        operators: &[AbstractOperator],
        _comparison_var_ids: &[usize],
    ) -> Self {
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

pub(super) fn decode_state_to_vectors(
    state_hash: usize,
    num_props: usize,
    domain_sizes: &[usize],
    numeric_domain_sizes: &[usize],
    hash_multipliers: &[usize],
    prop_out: &mut Vec<Vec<usize>>,
    num_out: &mut Vec<Vec<usize>>,
) {
    let mut props: Vec<usize> = Vec::with_capacity(num_props);
    for var_id in 0..num_props {
        let mult = hash_multipliers[var_id];
        let dom = domain_sizes[var_id];
        let val = (state_hash / mult) % dom;
        props.push(val);
    }
    let mut nums: Vec<usize> = Vec::with_capacity(numeric_domain_sizes.len());
    for (num_id, &dom_u) in numeric_domain_sizes.iter().enumerate() {
        let abs_var = abstraction_numeric_var(num_props, num_id);
        let mult = hash_multipliers[abs_var];
        let dom = dom_u;
        let part = (state_hash / mult) % dom;
        nums.push(part);
    }
    prop_out.push(props);
    num_out.push(nums);
}

impl DomainAbstractionFactory {
    /// Check the invariant the sparse propositional overlap depends on: a
    /// dimension the region does not list as constrained must admit its whole
    /// concrete domain. Violating it makes disjoint regions compare as
    /// overlapping, so an operator's cost can be claimed twice -- a wrong
    /// heuristic rather than a slow one.
    #[cfg(debug_assertions)]
    fn debug_assert_constrained_props_cover_narrowings(&self, region: &StateRegion) {
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

    pub(super) fn state_region_from_hash(
        &self,
        state_hash: usize,
        numeric_domain_sizes: &[usize],
        hash_multipliers: &[usize],
    ) -> Result<StateRegion> {
        let region = StateRegion::with_constrained_props(
            self.propositional_region_from_hash(state_hash, hash_multipliers)?,
            self.numeric_region_from_hash(state_hash, numeric_domain_sizes, hash_multipliers)?,
            Arc::clone(&self.state_constrained_props),
        );
        #[cfg(debug_assertions)]
        self.debug_assert_constrained_props_cover_narrowings(&region);
        Ok(region)
    }

    pub(super) fn state_region_from_facts(
        &self,
        task: &dyn AbstractNumericTask,
        facts: &[ExplicitFact],
    ) -> Result<StateRegion> {
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
                    self.concrete_values_for_abstract_value(fact.var(), fact.value())?;
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
                    .partition_interval(numeric_var_id, fact.value())
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
            StateRegion::with_constrained_props(propositions, numeric, constrained_props.into());
        #[cfg(debug_assertions)]
        self.debug_assert_constrained_props_cover_narrowings(&region);
        Ok(region)
    }

    pub(super) fn full_propositional_region(&self) -> Result<Vec<Vec<u32>>> {
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
            region.push((0..mapping.len() as u32).collect());
        }
        Ok(region)
    }

    pub(super) fn concrete_values_for_abstract_value(
        &self,
        var_id: usize,
        abstract_value: usize,
    ) -> Result<Vec<u32>> {
        // `filter_map().collect()` preallocates capacity matching the inner iterator's
        // upper-bound size_hint (here the full domain mapping length), so a var that
        // only has a handful of concrete values mapped to this abstract slot leaves
        // most of that capacity unused. Shrinking before returning saves typically
        // 50-90% of the per-inner-`Vec` heap allocation, which on SCP runs over many
        // state regions dominates the propositional representation.
        let mut values = self
            .domain_mapping
            .get(var_id)
            .with_context(|| format!("missing domain mapping for var {var_id}"))?
            .iter()
            .enumerate()
            .filter_map(|(concrete_value, &mapped_value)| {
                (mapped_value == abstract_value).then_some(concrete_value as u32)
            })
            .collect::<Vec<u32>>();
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
    ) -> Result<Vec<Vec<u32>>> {
        let mut region = Vec::with_capacity(self.domain_sizes.len());
        for (var_id, &domain_size) in self.domain_sizes.iter().enumerate() {
            ensure!(domain_size > 0, "domain size must be > 0 for var {var_id}");
            let multiplier = *hash_multipliers
                .get(var_id)
                .with_context(|| format!("missing hash multiplier for var {var_id}"))?;
            let abstract_value = (state_hash / multiplier) % domain_size;
            region.push(self.concrete_values_for_abstract_value(var_id, abstract_value)?);
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
            ensure!(
                domain_size > 0,
                "numeric domain size must be > 0 for var {numeric_var_id}"
            );
            let abs_var_id = abstraction_numeric_var(num_props, numeric_var_id);
            let multiplier = *hash_multipliers.get(abs_var_id).with_context(|| {
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

    pub(super) fn comparison_var_ids(&self) -> Vec<usize> {
        self.numeric_conditions
            .condition_var_ids()
            .filter(|&var_id| self.domain_sizes.get(var_id).copied().unwrap_or(1) > 1)
            .collect()
    }

    /// The task's goals in abstract values.
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
            let var = fact.var();
            if self.domain_sizes.get(var).copied().unwrap_or(1) <= 1 {
                continue;
            }
            let mapped = self
                .domain_mapping
                .get(var)
                .and_then(|mapping| mapping.get(fact.value()))
                .copied()
                .unwrap_or(fact.value());
            out.push(ExplicitFact::propositional(var, mapped));
        }

        out
    }

    pub fn is_goal_state(
        &self,
        state_hash: usize,
        goals: &[ExplicitFact],
        numeric_domain_sizes: &[usize],
        hash_multipliers: &[usize],
    ) -> bool {
        let num_props = self.domain_sizes.len();
        for g in goals {
            let var = g.var();
            let expected = g.value();
            let mult = hash_multipliers[var];
            let state = state_hash;
            let dom_size = if var < num_props {
                self.domain_sizes[var]
            } else {
                let n = var - num_props;
                numeric_domain_sizes.get(n).copied().unwrap_or(0)
            };
            let actual = (state / mult) % dom_size;
            if actual != expected {
                return false;
            }
        }
        true
    }

    pub(super) fn compute_initial_state_hash_determined(
        &self,
        task: &dyn AbstractNumericTask,
        numeric_domain_sizes: &[usize],
        hash_multipliers: &[usize],
        comparison_var_ids: &[usize],
    ) -> Result<usize> {
        let prop_init = task.get_initial_propositional_state_values();
        let num_init = task.get_initial_numeric_state_values();
        let num_props = self.domain_sizes.len();
        ensure!(
            prop_init.len() >= num_props,
            "initial propositional state too short: {} < {num_props}",
            prop_init.len()
        );
        ensure!(
            num_init.len() >= numeric_domain_sizes.len(),
            "initial numeric state too short: {} < {}",
            num_init.len(),
            numeric_domain_sizes.len()
        );

        let mut index: usize = 0;
        for var in 0..num_props {
            let mult = hash_multipliers[var];
            let concrete_value = if comparison_var_ids.contains(&var)
                && let Some(tree) = self.numeric_conditions.for_var(var)
            {
                ConditionValue::from(tree.evaluate_point(num_init)).as_usize()
            } else {
                prop_init[var]
            };
            let abs_val = *self.domain_mapping[var]
                .get(concrete_value)
                .with_context(|| {
                    format!(
                        "missing mapping for propositional var {var} value index {concrete_value}"
                    )
                })?;
            index += mult * abs_val;
        }

        for num_var_id in 0..numeric_domain_sizes.len() {
            let abs_var = abstraction_numeric_var(num_props, num_var_id);
            let mult = hash_multipliers[abs_var];
            let concrete_value = self
                .additive_numeric_views
                .get(num_var_id)
                .map(|view| view.evaluate(num_init))
                .unwrap_or(num_init[num_var_id]);
            let val = float_tolerance::canonicalize(concrete_value);
            ensure!(
                val.is_finite() && !val.is_nan(),
                "initial numeric value for var {num_var_id} must be finite, got {val}"
            );
            let parts = self
                .partitions
                .partitions(num_var_id)
                .with_context(|| format!("missing partitions for numeric var {num_var_id}"))?;
            let part = utils::partition_for_value(parts, val).with_context(|| {
                format!(
                    "initial numeric value {val} not contained in any partition for numeric var {num_var_id}"
                )
            })?;
            index += mult * part;
        }

        Ok(index)
    }

    /// The abstract class a comparison digit is cleared to before the verdict
    /// for this abstract state is derived: the class of [`ConditionValue::False`],
    /// which is also the class every abstract operator's comparison effect
    /// targets.
    pub(super) fn cleared_comparison_class(&self, var_id: usize) -> Result<usize> {
        self.domain_mapping[var_id]
            .get(ConditionValue::False.as_usize())
            .copied()
            .with_context(|| format!("missing FALSE mapping for comparison var {var_id}"))
    }

    /// Clear every comparison digit of `state_hash` to its
    /// [cleared class](Self::cleared_comparison_class), except the ones
    /// `fixed_comparisons` pins to a value.
    pub(super) fn clear_comparison_vars_except(
        &self,
        state_hash: usize,
        hash_multipliers: &[usize],
        comparison_var_ids: &[usize],
        fixed_comparisons: &[ExplicitFact],
    ) -> Result<usize> {
        let mut out = state_hash;
        for &var_id in comparison_var_ids {
            ensure!(
                var_id < self.domain_sizes.len(),
                "comparison var id out of range: {var_id}"
            );
            if self.domain_sizes[var_id] <= 1 {
                continue;
            }
            let mult = hash_multipliers[var_id];
            let dom = self.domain_sizes[var_id];
            ensure!(dom > 0, "domain size must be > 0 for var {var_id}");
            let cur = (out / mult) % dom;
            let target_abs = if let Some(fixed_value) = fixed_comparisons
                .iter()
                .find(|fact| fact.var() == var_id)
                .map(|fact| fact.value())
            {
                ensure!(
                    fixed_value < dom,
                    "fixed comparison value {fixed_value} out of abstract domain for var {var_id} with size {dom}"
                );
                fixed_value
            } else {
                self.cleared_comparison_class(var_id)?
            };
            let cur_offset = cur
                .checked_mul(mult)
                .context("comparison current digit offset overflow")?;
            let target_offset = target_abs
                .checked_mul(mult)
                .context("comparison target digit offset overflow")?;
            out = out
                .checked_sub(cur_offset)
                .context("comparison reset encountered an invalid state hash")?;
            out = out
                .checked_add(target_offset)
                .context("comparison reset hash overflow")?;
        }
        Ok(out)
    }

    #[allow(unused)]
    pub(super) fn build_numeric_intervals(
        &self,
        state_hash: usize,
        numeric_domain_sizes: &[usize],
        hash_multipliers: &[usize],
        task: &dyn AbstractNumericTask,
    ) -> Result<Vec<Interval>> {
        prepare_comparison_tree_inputs_from_abstract_state(
            task,
            self.numeric_conditions.all(),
            &self.partitions,
            AbstractStateHash {
                hash: state_hash,
                num_props: self.domain_sizes.len(),
                numeric_domain_sizes,
                hash_multipliers,
            },
        )
    }

    pub(super) fn enumerate_states_with_evaluated_comparisons(
        &self,
        base_state_hash: usize,
        task: &dyn AbstractNumericTask,
        numeric_domain_sizes: &[usize],
        hash_multipliers: &[usize],
        comparison_var_ids: &[usize],
        fixed_comparisons: &[ExplicitFact],
    ) -> Result<Vec<usize>> {
        if comparison_var_ids.is_empty() {
            return Ok(vec![base_state_hash]);
        }
        let num_props = self.domain_sizes.len();
        let cleared_state = self.clear_comparison_vars_except(
            base_state_hash,
            hash_multipliers,
            comparison_var_ids,
            fixed_comparisons,
        )?;

        // `fixed_comparisons` is typically empty or has 1-3 entries — replace the
        // per-call `HashMap<usize, usize>` (with default SipHash + heap alloc) with
        // a stack-friendly slice scan.
        let is_fixed_var =
            |var_id: usize| -> bool { fixed_comparisons.iter().any(|f| f.var() == var_id) };
        let is_evaluated_var = |var_id: usize| -> bool { comparison_var_ids.contains(&var_id) };

        // Build the numeric intervals for this abstract state ONCE, then
        // evaluate each comparison tree against the shared buffer. The old
        // path called `evaluate_comparison_tree_from_abstract_state` per tree,
        // each call allocating a fresh `Vec<Interval>` and re-running
        // `fill_derived_numeric_intervals_from_comparison_trees`.
        let mut numeric_intervals: Vec<Interval> = Vec::new();
        let mut intervals_built = false;

        let mut states: Vec<usize> = vec![cleared_state];
        for tree in self.numeric_conditions.iter() {
            let var_id = tree.prop_var_id();
            ensure!(
                var_id < num_props,
                "numeric condition variable out of range: {var_id} >= {num_props}"
            );
            if !is_evaluated_var(var_id) {
                continue;
            }
            if self.domain_sizes[var_id] <= 1 {
                continue;
            }
            if is_fixed_var(var_id) {
                continue;
            }

            // The digit starts at the class of `False`, so only `True` moves it.
            let mult = hash_multipliers[var_id];
            let delta_true = (self.domain_mapping[var_id]
                .get(ConditionValue::True.as_usize())
                .copied()
                .with_context(|| format!("missing TRUE mapping for comparison var {var_id}"))?
                as i32
                - self.cleared_comparison_class(var_id)? as i32)
                * mult as i32;

            if !intervals_built {
                prepare_comparison_tree_inputs_from_abstract_state_into(
                    task,
                    self.numeric_conditions.all(),
                    &self.partitions,
                    AbstractStateHash {
                        hash: base_state_hash,
                        num_props,
                        numeric_domain_sizes,
                        hash_multipliers,
                    },
                    &mut numeric_intervals,
                )?;
                intervals_built = true;
            }

            match tree.evaluate_interval(&numeric_intervals) {
                Some(true) => {
                    for s in &mut states {
                        *s = (*s as i32 + delta_true) as usize;
                    }
                }
                // The digit already stands at the class of `False`.
                Some(false) => {}
                // The abstract state straddles the comparison, so both verdicts
                // are reachable from it and both states have to be enumerated.
                None => {
                    let mut next: Vec<usize> = Vec::with_capacity(states.len() * 2);
                    for &s in &states {
                        next.push((s as i32 + delta_true) as usize);
                        next.push(s);
                    }
                    states = next;
                }
            }
        }
        Ok(states)
    }

    pub(super) fn enumerate_states_with_evaluated_comparisons_cached<'a>(
        &self,
        base_state_hash: usize,
        task: &dyn AbstractNumericTask,
        layout: ComparisonBranchingLayout<'_>,
        fixed_comparisons: &[ExplicitFact],
        memo: &'a mut ComparisonEnumerationMemo,
    ) -> Result<&'a [usize]> {
        let key = comparison_enumeration_signature(base_state_hash, fixed_comparisons);
        if memo.cache.contains_key(&key) {
            let states = memo
                .cache
                .get(&key)
                .expect("comparison enumeration cache key disappeared");
            return Ok(states.as_slice());
        }

        let states = self.enumerate_states_with_evaluated_comparisons(
            base_state_hash,
            task,
            layout.numeric_domain_sizes,
            layout.hash_multipliers,
            layout.comparison_var_ids,
            fixed_comparisons,
        )?;
        if memo.cache.len() < COMPARISON_ENUMERATION_CACHE_MAX_ENTRIES
            && memo.cached_state_count + states.len() <= COMPARISON_ENUMERATION_CACHE_MAX_STATES
        {
            memo.cached_state_count += states.len();
            memo.cache.insert(key, states);
            let states = memo
                .cache
                .get(&key)
                .expect("inserted comparison enumeration cache entry missing");
            return Ok(states.as_slice());
        }
        memo.overflow.clear();
        memo.overflow.extend_from_slice(&states);
        Ok(memo.overflow.as_slice())
    }
}
