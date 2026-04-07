#[cfg(test)]
mod tests;

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};

use anyhow::{Context, Result, anyhow, bail, ensure};
use ordered_float::NotNan;

use planners_sas::numeric::numeric_task::{AbstractNumericTask, ExplicitFact};

use super::abstract_operator_generator::{
    AbstractOperator, AbstractOperatorGenerator, DomainMapping,
};
use super::comparison_expression::{ComparisonTree, Interval};
use super::domain_abstraction::{ComparisonAxiomIndex, NumericPartitions};
use super::numeric_context::{
    evaluate_comparison_tree_from_abstract_state, evaluate_comparison_tree_from_initial_state,
    prepare_comparison_tree_inputs_from_abstract_state,
};
use super::utils;

const COMPARISON_TRUE_VAL: usize = 0;
const COMPARISON_FALSE_VAL: usize = 1;
const COMPARISON_UNKNOWN_VAL: usize = 2;

#[derive(Debug, Clone, Default)]
struct MatchTreeNode {
    value_children: HashMap<usize, Box<MatchTreeNode>>,
    wildcard_child: Option<Box<MatchTreeNode>>,
    ops: Vec<usize>,
}

#[derive(Debug, Clone)]
struct MatchTree {
    var_order: Vec<usize>,
    domain_sizes: Vec<usize>,
    numeric_domain_sizes: Vec<usize>,
    hash_multipliers: Vec<usize>,
    root: MatchTreeNode,
}

impl MatchTree {
    fn build(
        domain_sizes: &[usize],
        numeric_domain_sizes: &[usize],
        hash_multipliers: &[usize],
        operators: &[AbstractOperator],
        _comparison_var_ids: &[usize],
    ) -> Self {
        let num_props = domain_sizes.len();
        let mut vars_seen: HashSet<usize> = HashSet::new();
        for op in operators {
            for f in op.regression_preconditions.iter() {
                let domain_size = if f.var < num_props {
                    domain_sizes.get(f.var).copied().unwrap_or(0)
                } else {
                    let numeric_var = f.var - num_props;
                    numeric_domain_sizes.get(numeric_var).copied().unwrap_or(0)
                };
                if domain_size > 1 {
                    vars_seen.insert(f.var);
                }
            }
        }
        let mut var_order: Vec<usize> = vars_seen.into_iter().collect();
        var_order.sort_unstable();

        let mut tree = Self {
            var_order,
            domain_sizes: domain_sizes.to_vec(),
            numeric_domain_sizes: numeric_domain_sizes.to_vec(),
            hash_multipliers: hash_multipliers.to_vec(),
            root: MatchTreeNode::default(),
        };

        for (op_id, op) in operators.iter().enumerate() {
            let mut conds: HashMap<usize, usize> = HashMap::new();
            for f in op.regression_preconditions.iter() {
                conds.insert(f.var, f.value);
            }
            tree.insert(op_id, &conds);
        }

        tree
    }

    fn insert(&mut self, op_id: usize, conds: &HashMap<usize, usize>) {
        fn insert_rec(
            node: &mut MatchTreeNode,
            depth: usize,
            var_order: &[usize],
            conds: &HashMap<usize, usize>,
            op_id: usize,
        ) {
            if depth == var_order.len() {
                node.ops.push(op_id);
                return;
            }
            let var = var_order[depth];
            if let Some(&val) = conds.get(&var) {
                let child = node
                    .value_children
                    .entry(val)
                    .or_insert_with(|| Box::new(MatchTreeNode::default()));
                insert_rec(child.as_mut(), depth + 1, var_order, conds, op_id);
            } else {
                let child = node
                    .wildcard_child
                    .get_or_insert_with(|| Box::new(MatchTreeNode::default()));
                insert_rec(child.as_mut(), depth + 1, var_order, conds, op_id);
            }
        }

        insert_rec(&mut self.root, 0, &self.var_order, conds, op_id);
    }

    fn get_applicable_operator_ids(&self, state_hash: usize, out: &mut Vec<usize>) {
        out.clear();
        if self.var_order.is_empty() {
            out.extend_from_slice(&self.root.ops);
            return;
        }
        let mut stack: Vec<(&MatchTreeNode, usize)> = Vec::new();
        stack.push((&self.root, 0));
        while let Some((node, depth)) = stack.pop() {
            if depth == self.var_order.len() {
                out.extend_from_slice(&node.ops);
                continue;
            }
            let var = self.var_order[depth];
            let actual = self.get_var_value(state_hash, var);
            if let Some(child) = node.wildcard_child.as_deref() {
                stack.push((child, depth + 1));
            }
            if let Some(child) = node.value_children.get(&actual) {
                stack.push((child.as_ref(), depth + 1));
            }
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

#[derive(Debug, Clone)]
pub struct AbstractDistanceTable {
    pub distances: Vec<f64>,
    // Per-state operator leading to a goal along a shortest path.
    pub generating_op_ids: Vec<Option<usize>>,
    pub initial_state_hash: usize,
    pub goal_facts: Vec<ExplicitFact>,
    pub hash_multipliers: Vec<usize>,
    pub numeric_domain_sizes: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct WildcardPlanResult {
    // Per-step set of concrete operator IDs
    pub wildcard_plan: Vec<Vec<usize>>,
    // Path of abstract state hashes (`len = steps+1`).
    pub abstract_state_hashes: Vec<usize>,
    // Decoded propositional values along path.
    pub abstract_prop_states: Vec<Vec<usize>>,
    // Decoded numeric partitions along path.
    pub abstract_numeric_states: Vec<Vec<usize>>,
}

#[derive(Debug, Clone)]
pub struct DomainAbstractionFactory {
    domain_mapping: DomainMapping,
    domain_sizes: Vec<usize>,
    partitions: NumericPartitions,
    numeric_domain_sizes: Vec<usize>,
    comparison_index: Option<ComparisonAxiomIndex>,
    comparison_trees: Vec<ComparisonTree>,
}

impl DomainAbstractionFactory {
    pub fn new(
        task: &dyn AbstractNumericTask,
        domain_mapping: DomainMapping,
        domain_sizes: Vec<usize>,
        partitions: NumericPartitions,
        numeric_domain_sizes: Vec<usize>,
    ) -> Result<Self> {
        ensure!(
            domain_mapping.len() == domain_sizes.len(),
            "domain_mapping/domain_sizes length mismatch"
        );
        for (var, &abs_size) in domain_sizes.iter().enumerate() {
            ensure!(
                abs_size > 0,
                "non-positive abstract domain size for var {var}: {abs_size}"
            );

            let concrete_size = task
                .get_variable_domain_size(var)
                .map_err(|e| anyhow!(e.to_string()))
                .with_context(|| format!("get_variable_domain_size({var}) failed"))?;
            ensure!(
                concrete_size > 0,
                "non-positive concrete domain size for var {var}: {concrete_size}"
            );
            ensure!(
                abs_size <= concrete_size,
                "abstract domain size for var {var} exceeds concrete size ({abs_size} > {concrete_size})"
            );

            ensure!(
                domain_mapping[var].len() == concrete_size,
                "domain_mapping[{var}] has len {}, expected concrete size {concrete_size}",
                domain_mapping[var].len()
            );

            for (val, &mapped) in domain_mapping[var].iter().enumerate() {
                ensure!(
                    mapped < abs_size,
                    "domain_mapping[{var}][{val}]={mapped} out of range for abstract size {abs_size}"
                );
            }
        }
        for (n, &parts) in numeric_domain_sizes.iter().enumerate() {
            ensure!(parts > 0, "numeric_domain_sizes[{n}] must be > 0");
            let actual = partitions.partitions(n).map(|p| p.len()).unwrap_or(0);
            ensure!(
                actual == parts,
                "numeric_domain_sizes[{n}]={parts} does not match partitions len {actual}"
            );
        }

        let comparison_index = if task.comparison_axioms().is_empty() {
            None
        } else {
            Some(
                ComparisonAxiomIndex::from_task(task)
                    .map_err(|e| anyhow!(e))
                    .context("failed to build ComparisonAxiomIndex")?,
            )
        };

        let mut comparison_trees: Vec<ComparisonTree> =
            Vec::with_capacity(task.comparison_axioms().len());
        for comparison_axiom_id in 0..task.comparison_axioms().len() {
            let tree = ComparisonTree::from_task(task, comparison_axiom_id).map_err(|e| {
                anyhow!(
                    "failed to build ComparisonTree for comparison axiom {comparison_axiom_id}: {e:?}"
                )
            })?;
            comparison_trees.push(tree);
        }

        Ok(Self {
            domain_mapping,
            domain_sizes,
            partitions,
            numeric_domain_sizes,
            comparison_index,
            comparison_trees,
        })
    }

    pub fn partitions(&self) -> &NumericPartitions {
        &self.partitions
    }

    pub fn domain_mapping(&self) -> &DomainMapping {
        &self.domain_mapping
    }

    pub fn domain_sizes(&self) -> &[usize] {
        &self.domain_sizes
    }

    pub fn numeric_domain_sizes(&self) -> &[usize] {
        &self.numeric_domain_sizes
    }

    pub fn comparison_index(&self) -> Option<&ComparisonAxiomIndex> {
        self.comparison_index.as_ref()
    }

    pub fn comparison_trees(&self) -> &[ComparisonTree] {
        &self.comparison_trees
    }

    pub fn make_operator_generator(
        &self,
        task: &dyn AbstractNumericTask,
        combine_labels: bool,
    ) -> Result<AbstractOperatorGenerator> {
        AbstractOperatorGenerator::new(
            task,
            self.domain_mapping.clone(),
            self.domain_sizes.clone(),
            self.partitions.clone(),
            self.numeric_domain_sizes.clone(),
            combine_labels,
        )
    }

    /// Runs numeric-fd style implicit regression Dijkstra and returns distances-to-goal for
    /// all abstract states plus the generating operator per state.
    pub fn build_abstract_distance_table(
        &self,
        task: &dyn AbstractNumericTask,
        combine_labels: bool,
        dump_distances: bool,
    ) -> Result<AbstractDistanceTable> {
        let mut generator = self.make_operator_generator(task, combine_labels)?;
        let operators = generator.build_abstract_operators(task)?;
        self.build_distance_table_with_operators(task, &generator, &operators, dump_distances)
    }

    /// Computes an abstract wildcard plan (sequence of per-step concrete-op-ID sets) by:
    /// 1) Computing abstract goal distances with implicit regression Dijkstra.
    /// 2) Extracting a shortest-path abstract plan from the initial abstract state.
    /// 3) Collecting all cheapest realizations per step.
    pub fn compute_wildcard_plan(
        &self,
        task: &dyn AbstractNumericTask,
        combine_labels: bool,
        dump_distances: bool,
    ) -> Result<Option<WildcardPlanResult>> {
        let mut generator = self.make_operator_generator(task, combine_labels)?;
        let operators = generator.build_abstract_operators(task)?;
        let table =
            self.build_distance_table_with_operators(task, &generator, &operators, dump_distances)?;

        let comparison_var_ids = self.comparison_var_ids();
        let match_tree = MatchTree::build(
            generator.domain_sizes(),
            generator.numeric_domain_sizes(),
            generator.hash_multipliers(),
            &operators,
            &comparison_var_ids,
        );

        self.compute_wildcard_plan_from_table(
            task,
            &generator,
            &operators,
            &table,
            &comparison_var_ids,
            &match_tree,
        )
    }

    fn build_distance_table_with_operators(
        &self,
        task: &dyn AbstractNumericTask,
        generator: &AbstractOperatorGenerator,
        operators: &[AbstractOperator],
        dump_distances: bool,
    ) -> Result<AbstractDistanceTable> {
        let hash_multipliers = generator.hash_multipliers();
        let numeric_domain_sizes = generator.numeric_domain_sizes();
        let comparison_var_ids = self.comparison_var_ids();

        let goal_facts = self.compute_abstract_goals(task);

        // Numeric-fd computes a *single* initial abstract state hash directly from the
        // concrete initial state (comparisons are evaluated, not enumerated).
        let init_hash = self.compute_initial_state_hash_determined(
            task,
            numeric_domain_sizes,
            hash_multipliers,
            &comparison_var_ids,
        )?;

        let num_states = compute_num_states(&self.domain_sizes, numeric_domain_sizes)?;

        let match_tree = MatchTree::build(
            &self.domain_sizes,
            numeric_domain_sizes,
            hash_multipliers,
            operators,
            &comparison_var_ids,
        );

        let (distances, generating_op_ids) = self.compute_distances_and_generating_ops(
            task,
            operators,
            &match_tree,
            &goal_facts,
            init_hash,
            numeric_domain_sizes,
            hash_multipliers,
            &comparison_var_ids,
            num_states,
        )?;

        let table = AbstractDistanceTable {
            distances,
            generating_op_ids,
            initial_state_hash: init_hash,
            goal_facts,
            hash_multipliers: hash_multipliers.to_vec(),
            numeric_domain_sizes: numeric_domain_sizes.to_vec(),
        };

        if dump_distances {
            self.dump_distances(task, &table);
        }

        Ok(table)
    }

    /// Prints a numeric-fd style table of core variables for all reachable abstract states.
    ///
    /// Core variables are:
    /// - all numeric variables with more than one partition,
    /// - all non-axiom propositional variables with abstract domain size > 1.
    pub fn dump_distances(&self, task: &dyn AbstractNumericTask, table: &AbstractDistanceTable) {
        utils::dump_distances(self, task, table);
    }
    fn comparison_var_ids(&self) -> Vec<usize> {
        self.comparison_trees
            .iter()
            .map(|t| t.affected_var_id)
            .collect()
    }

    fn compute_abstract_goals(&self, task: &dyn AbstractNumericTask) -> Vec<ExplicitFact> {
        use planners_sas::numeric::numeric_task::ExplicitFact;

        let mut goal_axiom_map: HashMap<usize, usize> = HashMap::new();
        for (idx, ax) in task.axioms().iter().enumerate() {
            if !ax.conditions().is_empty() {
                goal_axiom_map.insert(ax.var_id(), idx);
            }
        }

        let mut out: Vec<ExplicitFact> = Vec::new();
        for i in 0..task.get_num_goals() {
            let g = task.get_goal_fact(i);
            let var = g.var;
            if let Some(&ax_idx) = goal_axiom_map.get(&var) {
                let ax = &task.axioms()[ax_idx];
                for cond in ax.conditions() {
                    let v = cond.var;
                    if self.domain_sizes.get(v).copied().unwrap_or(1) <= 1 {
                        continue;
                    }
                    let val = cond.value;
                    let mapped = self
                        .domain_mapping
                        .get(v)
                        .and_then(|m| m.get(val))
                        .copied()
                        .unwrap_or(cond.value);
                    out.push(ExplicitFact::new(cond.var, mapped));
                }
            } else {
                let v = g.var;
                if self.domain_sizes.get(v).copied().unwrap_or(1) <= 1 {
                    continue;
                }
                let val = g.value;
                let mapped = self
                    .domain_mapping
                    .get(v)
                    .and_then(|m| m.get(val))
                    .copied()
                    .unwrap_or(g.value);
                out.push(ExplicitFact::new(g.var, mapped));
            }
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
            let var = g.var;
            let expected = g.value;
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

    fn compute_initial_state_hash_determined(
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

        let comparison_var_ids: HashSet<usize> = comparison_var_ids.iter().copied().collect();
        let mut index: usize = 0;
        for var in 0..num_props {
            let mult = hash_multipliers[var];
            let concrete_val = if comparison_var_ids.contains(&var) {
                if let Some(tree) = self
                    .comparison_trees
                    .iter()
                    .find(|tree| tree.affected_var_id == var)
                {
                    match evaluate_comparison_tree_from_initial_state(task, tree)? {
                        Some(true) => COMPARISON_TRUE_VAL,
                        Some(false) => COMPARISON_FALSE_VAL,
                        None => prop_init[var],
                    }
                } else {
                    prop_init[var]
                }
            } else {
                prop_init[var]
            };
            let abs_val = *self.domain_mapping[var]
                .get(concrete_val)
                .with_context(|| {
                    format!(
                        "missing mapping for propositional var {var} value index {concrete_val}"
                    )
                })?;
            index += mult * abs_val;
        }

        for num_var_id in 0..numeric_domain_sizes.len() {
            let abs_var = num_props + num_var_id;
            let mult = hash_multipliers[abs_var];
            let val = num_init[num_var_id];
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

    fn reset_comparison_vars_to_unknown_except(
        &self,
        state_hash: usize,
        hash_multipliers: &[usize],
        comparison_var_ids: &[usize],
        fixed_comparisons: &[ExplicitFact],
    ) -> Result<usize> {
        let mut fixed: HashSet<usize> = HashSet::new();
        for f in fixed_comparisons {
            fixed.insert(f.var);
        }

        let mut out = state_hash;
        for &var_id in comparison_var_ids {
            ensure!(
                var_id < self.domain_sizes.len(),
                "comparison var id out of range: {var_id}"
            );
            if fixed.contains(&(var_id)) {
                continue;
            }
            if self.domain_sizes[var_id] <= 1 {
                continue;
            }
            let mult = hash_multipliers[var_id];
            let dom = self.domain_sizes[var_id];
            ensure!(dom > 0, "domain size must be > 0 for var {var_id}");
            let cur = (out / mult) % dom;
            let unknown_abs = *self.domain_mapping[var_id]
                .get(COMPARISON_UNKNOWN_VAL)
                .with_context(|| format!("missing UNKNOWN mapping for comparison var {var_id}"))?;
            out += (unknown_abs - cur) * mult;
        }
        Ok(out)
    }

    #[allow(unused)]
    fn build_numeric_intervals(
        &self,
        state_hash: usize,
        numeric_domain_sizes: &[usize],
        hash_multipliers: &[usize],
        task: &dyn AbstractNumericTask,
    ) -> Result<Vec<Interval>> {
        prepare_comparison_tree_inputs_from_abstract_state(
            task,
            &self.comparison_trees,
            &self.partitions,
            state_hash,
            self.domain_sizes.len(),
            numeric_domain_sizes,
            hash_multipliers,
        )
    }

    fn enumerate_states_with_evaluated_comparisons(
        &self,
        base_state_hash: usize,
        task: &dyn AbstractNumericTask,
        numeric_domain_sizes: &[usize],
        hash_multipliers: &[usize],
        comparison_var_ids: &[usize],
        fixed_comparisons: &[ExplicitFact],
    ) -> Result<Vec<usize>> {
        let num_props = self.domain_sizes.len();
        let state_unknown = self.reset_comparison_vars_to_unknown_except(
            base_state_hash,
            hash_multipliers,
            comparison_var_ids,
            fixed_comparisons,
        )?;

        let mut fixed_map: HashMap<usize, usize> = HashMap::new();
        for f in fixed_comparisons {
            fixed_map.insert(f.var, f.value);
        }

        let mut states: Vec<usize> = vec![state_unknown];
        for tree in &self.comparison_trees {
            let var_id = tree.affected_var_id;
            ensure!(
                var_id < num_props,
                "comparison tree affected_var_id out of range: {var_id} >= {num_props}"
            );
            if self.domain_sizes[var_id] <= 1 {
                continue;
            }
            if fixed_map.contains_key(&var_id) {
                continue;
            }

            let mult = hash_multipliers[var_id];
            let unknown_abs = *self.domain_mapping[var_id]
                .get(COMPARISON_UNKNOWN_VAL)
                .with_context(|| format!("missing UNKNOWN mapping for comparison var {var_id}"))?
                as i32;
            let delta_true = (self.domain_mapping[var_id]
                .get(COMPARISON_TRUE_VAL)
                .copied()
                .with_context(|| format!("missing TRUE mapping for comparison var {var_id}"))?
                as i32
                - unknown_abs)
                * mult as i32;
            let delta_false = (self.domain_mapping[var_id]
                .get(COMPARISON_FALSE_VAL)
                .copied()
                .with_context(|| format!("missing FALSE mapping for comparison var {var_id}"))?
                as i32
                - unknown_abs)
                * mult as i32;

            match evaluate_comparison_tree_from_abstract_state(
                task,
                tree,
                &self.partitions,
                base_state_hash,
                num_props,
                numeric_domain_sizes,
                hash_multipliers,
            )? {
                Some(true) => {
                    for s in &mut states {
                        *s = (*s as i32 + delta_true) as usize;
                    }
                }
                Some(false) => {
                    for s in &mut states {
                        *s = (*s as i32 + delta_false) as usize;
                    }
                }
                None => {
                    let mut next: Vec<usize> = Vec::with_capacity(states.len() * 2);
                    for &s in &states {
                        next.push((s as i32 + delta_true) as usize);
                        next.push((s as i32 + delta_false) as usize);
                    }
                    states = next;
                }
            }
        }
        Ok(states)
    }

    fn compute_wildcard_plan_from_table(
        &self,
        task: &dyn AbstractNumericTask,
        generator: &AbstractOperatorGenerator,
        operators: &[AbstractOperator],
        table: &AbstractDistanceTable,
        comparison_var_ids: &[usize],
        match_tree: &MatchTree,
    ) -> Result<Option<WildcardPlanResult>> {
        let domain_sizes = generator.domain_sizes();
        let hash_multipliers = generator.hash_multipliers();
        let num_props = domain_sizes.len();
        let numeric_domain_sizes = generator.numeric_domain_sizes();

        let dist = &table.distances;
        let generating_op = &table.generating_op_ids;

        let mut current_hash = table.initial_state_hash;
        if current_hash >= dist.len() || !dist[current_hash].is_finite() {
            return Ok(None);
        }

        let mut wildcard_plan: Vec<Vec<usize>> = Vec::new();
        let mut abstract_state_hashes: Vec<usize> = vec![current_hash];
        let mut seen_states: Vec<usize> = Vec::new();

        // For debugging / parity with numeric-fd deviation code.
        let mut abstract_prop_states: Vec<Vec<usize>> = Vec::new();
        let mut abstract_numeric_states: Vec<Vec<usize>> = Vec::new();
        decode_state_to_vectors(
            current_hash,
            num_props,
            domain_sizes,
            numeric_domain_sizes,
            hash_multipliers,
            &mut abstract_prop_states,
            &mut abstract_numeric_states,
        );

        let mut safety_steps = 0usize;
        while !self.is_goal_state(
            current_hash,
            &table.goal_facts,
            numeric_domain_sizes,
            hash_multipliers,
        ) {
            safety_steps += 1;
            if safety_steps > dist.len() + 1 {
                bail!("abstract plan extraction exceeded safety limit")
            }
            let Some(op_id) = generating_op.get(current_hash).copied().flatten() else {
                bail!("missing generating operator for state {current_hash} with finite distance");
            };
            let op = operators
                .get(op_id)
                .with_context(|| format!("generating op id out of bounds: {op_id}"))?;

            // Recompute successors on-the-fly like numeric-fd.
            let candidate_hash_effect = op.hash_effect;
            let base_successor = (current_hash as i32).wrapping_sub(candidate_hash_effect) as usize;

            // Reset all comparison vars to UNKNOWN (no fixed comparisons here).
            let base_successor = self.reset_comparison_vars_to_unknown_except(
                base_successor,
                hash_multipliers,
                comparison_var_ids,
                &[],
            )?;

            let possible_successors = self.enumerate_states_with_evaluated_comparisons(
                base_successor,
                task,
                numeric_domain_sizes,
                hash_multipliers,
                comparison_var_ids,
                &[],
            )?;

            let cur_d = dist[current_hash];
            ensure!(cur_d.is_finite(), "current distance must be finite");
            let mut chosen_successor: Option<usize> = None;
            let mut lowest_so_far = cur_d;
            for &cand in &possible_successors {
                if cand == current_hash {
                    continue;
                }
                if seen_states.contains(&cand) {
                    continue;
                }
                if cand >= dist.len() {
                    continue;
                }
                let cd = dist[cand];
                if !cd.is_finite() {
                    continue;
                }
                let valid_progress =
                    (cd < cur_d && op.cost > 0.0) || ((cd - cur_d).abs() <= 1e-9 && op.cost == 0.0);
                if valid_progress && chosen_successor.is_none_or(|x| cand > x) {
                    chosen_successor = Some(cand);
                    lowest_so_far = cd;
                }
            }
            let successor_hash = chosen_successor.with_context(|| {
                format!(
                    "no successor satisfies dist equation for state {current_hash} with op {op_id}"
                )
            })?;
            ensure!(
                successor_hash < dist.len(),
                "successor hash out of range: {successor_hash}"
            );
            ensure!(
                (lowest_so_far - cur_d + op.cost).abs() <= 1e-6,
                "chosen successor violates C++ plan-extraction distance relation"
            );
            let required_cost = op.cost;

            // Collect cheapest concrete ops like numeric-fd does: from the first
            // matching abstract operator group only.
            let mut step: Vec<usize> = Vec::new();
            let mut applicable_operator_ids: Vec<usize> = Vec::new();
            match_tree.get_applicable_operator_ids(base_successor, &mut applicable_operator_ids);
            for &cand_op_id in &applicable_operator_ids {
                let cand_op = operators
                    .get(cand_op_id)
                    .with_context(|| format!("candidate op id out of bounds: {cand_op_id}"))?;
                if (cand_op.cost - required_cost).abs() > 1e-9 {
                    continue;
                }
                let base_predecessor =
                    (base_successor as i32).wrapping_add(cand_op.hash_effect) as usize;
                let fixed_comparisons = get_comparison_preconditions(cand_op, comparison_var_ids);
                let possible_predecessors = self.enumerate_states_with_evaluated_comparisons(
                    base_predecessor,
                    task,
                    numeric_domain_sizes,
                    hash_multipliers,
                    comparison_var_ids,
                    &fixed_comparisons,
                )?;
                if possible_predecessors.contains(&current_hash) {
                    step = cand_op.concrete_op_ids.clone();
                    step.sort_unstable();
                    step.dedup();
                    break;
                }
            }
            wildcard_plan.push(step);

            seen_states.push(current_hash);
            current_hash = successor_hash;
            abstract_state_hashes.push(current_hash);
            decode_state_to_vectors(
                current_hash,
                num_props,
                domain_sizes,
                numeric_domain_sizes,
                hash_multipliers,
                &mut abstract_prop_states,
                &mut abstract_numeric_states,
            );
        }

        Ok(Some(WildcardPlanResult {
            wildcard_plan,
            abstract_state_hashes,
            abstract_prop_states,
            abstract_numeric_states,
        }))
    }

    #[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
    fn compute_distances_and_generating_ops(
        &self,
        task: &dyn AbstractNumericTask,
        operators: &[AbstractOperator],
        match_tree: &MatchTree,
        goal_facts: &[ExplicitFact],
        initial_state_hash: usize,
        numeric_domain_sizes: &[usize],
        hash_multipliers: &[usize],
        comparison_var_ids: &[usize],
        num_states: usize,
    ) -> Result<(Vec<f64>, Vec<Option<usize>>)> {
        let num_props = self.domain_sizes.len();
        let mut distances: Vec<f64> = vec![f64::INFINITY; num_states];
        let mut generating_op_ids: Vec<Option<usize>> = vec![None; num_states];

        let mut core_vars: Vec<u32> = Vec::new();
        for (v, &dom) in self.domain_sizes.iter().enumerate() {
            if dom > 1 {
                core_vars.push(v as u32);
            }
        }
        for (n, &dom) in numeric_domain_sizes.iter().enumerate() {
            if dom > 1 {
                core_vars.push((num_props + n) as u32);
            }
        }
        core_vars.sort_unstable();
        core_vars.dedup();

        let mut heap: BinaryHeap<(Reverse<NotNan<f64>>, usize)> = BinaryHeap::new();

        // Initialize with feasible goal states.
        for state_hash in 0..num_states {
            if !self.is_goal_state(
                state_hash,
                goal_facts,
                numeric_domain_sizes,
                hash_multipliers,
            ) {
                continue;
            }
            let alts = self.enumerate_states_with_evaluated_comparisons(
                state_hash,
                task,
                numeric_domain_sizes,
                hash_multipliers,
                comparison_var_ids,
                &[],
            )?;
            if !alts.contains(&state_hash) {
                continue;
            }
            distances[state_hash] = 0.0;
            heap.push((Reverse(NotNan::new(0.0).unwrap()), state_hash));
        }

        let mut applicable_operator_ids: Vec<usize> = Vec::new();
        while let Some((Reverse(d), state_hash)) = heap.pop() {
            let d = d.into_inner();
            if d > distances[state_hash] + 1e-12 {
                continue;
            }

            let base_state = self.reset_comparison_vars_to_unknown_except(
                state_hash,
                hash_multipliers,
                comparison_var_ids,
                &[],
            )?;

            match_tree.get_applicable_operator_ids(base_state, &mut applicable_operator_ids);
            for &op_id in &applicable_operator_ids {
                let op = &operators[op_id];
                ensure!(op.cost.is_finite(), "abstract operator cost must be finite");
                let alternative_cost = d + op.cost;
                let predecessor_base = (base_state as i32 + op.hash_effect) as usize;
                debug_assert!(
                    predecessor_base < num_states,
                    "[DA] predecessor base hash is out of bounds: {predecessor_base}"
                );
                // TODO: The next line should be impossible. Debug
                // if predecessor_base_i64 < 0 || predecessor_base_i64 >= num_states as i64 {
                //     eprintln!(
                //         "[DA_OOB] SKIPPED predecessor_base={predecessor_base_i64} num_states={num_states} base_state={base_state} hash_effect={}",
                //         op.hash_effect
                //     );
                //     continue;
                // }
                let fixed_comparisons = get_comparison_preconditions(op, comparison_var_ids);
                let possible_predecessors = self.enumerate_states_with_evaluated_comparisons(
                    predecessor_base,
                    task,
                    numeric_domain_sizes,
                    hash_multipliers,
                    comparison_var_ids,
                    &fixed_comparisons,
                )?;

                let representative_predecessor = possible_predecessors.iter().copied().max();

                for pred in possible_predecessors.iter().copied() {
                    debug_assert!(pred < num_states, "predecessor hash does not fit usize");

                    if alternative_cost + 1e-12 < distances[pred] {
                        distances[pred] = alternative_cost;
                        generating_op_ids[pred] = Some(op_id);
                        if pred == initial_state_hash || Some(pred) == representative_predecessor {
                            heap.push((
                                Reverse(
                                    NotNan::new(alternative_cost)
                                        .context("alternative cost is NaN")?,
                                ),
                                pred,
                            ));
                        }
                    }
                }
            }
        }

        Ok((distances, generating_op_ids))
    }
}

fn compute_num_states(domain_sizes: &[usize], numeric_domain_sizes: &[usize]) -> Result<usize> {
    let mut num: usize = 1;
    for (i, &s) in domain_sizes.iter().enumerate() {
        ensure!(s > 0, "domain size for var {i} must be > 0, got {s}");
        num = num
            .checked_mul(s)
            .context("abstract state space too large (overflow)")?;
    }
    for &s in numeric_domain_sizes.iter() {
        num = num
            .checked_mul(s)
            .context("abstract state space too large (overflow)")?;
    }
    Ok(num)
}

fn get_comparison_preconditions(
    op: &AbstractOperator,
    comparison_var_ids: &[usize],
) -> Vec<ExplicitFact> {
    let mut out: Vec<ExplicitFact> = Vec::new();
    for f in &op.preconditions {
        if comparison_var_ids.contains(&f.var) && f.value != COMPARISON_UNKNOWN_VAL {
            out.push(f.clone());
        }
    }
    out
}

fn decode_state_to_vectors(
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
        let abs_var = num_props + num_id;
        let mult = hash_multipliers[abs_var];
        let dom = dom_u;
        let part = (state_hash / mult) % dom;
        nums.push(part);
    }
    prop_out.push(props);
    num_out.push(nums);
}
