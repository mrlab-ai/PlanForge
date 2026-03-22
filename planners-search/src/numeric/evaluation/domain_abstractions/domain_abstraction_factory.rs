use std::cmp::Reverse;
use std::collections::{BTreeSet, BinaryHeap, HashMap, HashSet, VecDeque};

use ordered_float::NotNan;

use planners_sas::numeric::numeric_task::{AbstractNumericTask, NumericType};

use super::abstract_operator_generator::{AbstractOperator, AbstractOperatorGenerator, DomainMapping};
use super::comparison_expression::{ComparisonTree, Interval};
use super::domain_abstraction::{ComparisonAxiomIndex, NumericPartitions};

const COMPARISON_TRUE_VAL: i32 = 0;
const COMPARISON_FALSE_VAL: i32 = 1;
const COMPARISON_UNKNOWN_VAL: i32 = 2;

#[derive(Debug, Clone, Default)]
struct MatchTreeNode {
	value_children: HashMap<i32, Box<MatchTreeNode>>,
	wildcard_child: Option<Box<MatchTreeNode>>,
	ops: Vec<usize>,
}

#[derive(Debug, Clone)]
struct MatchTree {
	var_order: Vec<usize>,
	domain_sizes: Vec<i32>,
	numeric_domain_sizes: Vec<usize>,
	hash_multipliers: Vec<i32>,
	root: MatchTreeNode,
}

impl MatchTree {
	fn build(
		domain_sizes: &[i32],
		numeric_domain_sizes: &[usize],
		hash_multipliers: &[i32],
		operators: &[AbstractOperator],
		comparison_var_ids: &[usize],
	) -> Self {
		let mut freq: HashMap<usize, usize> = HashMap::new();
		for op in operators {
			for f in op.regression_preconditions.iter() {
				let Ok(var) = usize::try_from(f.var()) else {
					continue;
				};
				if comparison_var_ids.contains(&var) {
					continue;
				}
				*freq.entry(var).or_insert(0) += 1;
			}
		}
		let mut var_order: Vec<usize> = freq.keys().copied().collect();
		var_order.sort_by(|a, b| {
			let fa = freq.get(a).copied().unwrap_or(0);
			let fb = freq.get(b).copied().unwrap_or(0);
			fb.cmp(&fa).then_with(|| a.cmp(b))
		});

		let mut tree = Self {
			var_order,
			domain_sizes: domain_sizes.to_vec(),
			numeric_domain_sizes: numeric_domain_sizes.to_vec(),
			hash_multipliers: hash_multipliers.to_vec(),
			root: MatchTreeNode::default(),
		};

		for (op_id, op) in operators.iter().enumerate() {
			let mut conds: HashMap<usize, i32> = HashMap::new();
			for f in op.regression_preconditions.iter() {
				let Ok(var) = usize::try_from(f.var()) else {
					continue;
				};
				if comparison_var_ids.contains(&var) {
					continue;
				}
				conds.insert(var, f.value());
			}
			tree.insert(op_id, &conds);
		}

		tree
	}

	fn insert(&mut self, op_id: usize, conds: &HashMap<usize, i32>) {
		fn insert_rec(
			node: &mut MatchTreeNode,
			depth: usize,
			var_order: &[usize],
			conds: &HashMap<usize, i32>,
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

	fn get_applicable_operator_ids(&self, state_hash: i32, out: &mut Vec<usize>) {
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
			if let Some(child) = node.value_children.get(&actual) {
				stack.push((child.as_ref(), depth + 1));
			}
			if let Some(child) = node.wildcard_child.as_deref() {
				stack.push((child, depth + 1));
			}
		}
	}

	fn get_var_value(&self, state_hash: i32, var: usize) -> i32 {
		let num_props = self.domain_sizes.len();
		let mult = self.hash_multipliers.get(var).copied().unwrap_or(1) as i64;
		let state = state_hash as i64;
		let dom_size: i64 = if var < num_props {
			self.domain_sizes.get(var).copied().unwrap_or(0) as i64
		} else {
			let n = var - num_props;
			self.numeric_domain_sizes.get(n).copied().unwrap_or(0) as i64
		};
		if dom_size <= 0 {
			return 0;
		}
		((state / mult) % dom_size) as i32
	}
}

#[derive(Debug, Clone)]
pub struct AbstractTransition {
	pub op_id: usize,
	pub successor: usize,
	pub cost: f64,
}

#[derive(Debug, Clone)]
pub struct AbstractStateSpace {
	pub states: Vec<i32>,
	pub outgoing: Vec<Vec<AbstractTransition>>,
	pub incoming: Vec<Vec<(usize, usize, f64)>>, // (pred, op_id, cost)
	pub initial_state_ids: Vec<usize>,
	pub goal_facts: Vec<planners_sas::numeric::numeric_task::Fact>,
}

#[derive(Debug, Clone)]
pub struct WildcardPlanResult {
	pub wildcard_plan: Vec<Vec<usize>>, // per-step set of concrete operator IDs
	pub abstract_state_ids: Vec<usize>, // path of abstract node IDs (len = steps+1)
	pub abstract_prop_states: Vec<Vec<i32>>, // decoded propositional values along path
	pub abstract_numeric_states: Vec<Vec<i32>>, // decoded numeric partitions along path
}

#[derive(Debug, Clone)]
pub struct DomainAbstractionFactory {
	partitions: NumericPartitions,
	numeric_domain_sizes: Vec<usize>,
	comparison_index: Option<ComparisonAxiomIndex>,
	comparison_trees: Vec<ComparisonTree>,
}

impl DomainAbstractionFactory {
	/// Builds a factory that uses `Interval` partitions for numeric variables.
	///
	/// Current behavior:
	/// - Constants: singleton partitions at their initial value.
	/// - Regular vars: split at constant values that appear in any `ComparisonTree` that depends on
	///   the variable (directly or via derived expressions).
	/// - Derived/cost vars: unbounded single partition.
	pub fn from_task(task: &dyn AbstractNumericTask) -> Self {
		let comparison_index = ComparisonAxiomIndex::from_task(task).ok();

		let mut comparison_trees: Vec<ComparisonTree> = Vec::new();
		for comparison_axiom_id in 0..task.comparison_axioms().len() {
			if let Ok(tree) = ComparisonTree::from_task(task, comparison_axiom_id) {
				comparison_trees.push(tree);
			}
		}

		let initial_numeric_values = task.get_initial_numeric_state_values();
		let num_numeric_vars = task.numeric_variables().len();

		let mut cutpoints_by_var: Vec<BTreeSet<NotNan<f64>>> =
			vec![BTreeSet::new(); num_numeric_vars];
		for tree in &comparison_trees {
			let constant_values = constant_leaf_values(tree, task, &initial_numeric_values);
			if constant_values.is_empty() {
				continue;
			}

			for dep in tree.regular_numeric_var_dependencies(task) {
				let Ok(dep_idx) = usize::try_from(dep) else {
					continue;
				};
				if dep_idx >= cutpoints_by_var.len() {
					continue;
				}
				for &v in &constant_values {
					let Ok(v) = NotNan::new(v) else {
						continue;
					};
					if v.is_finite() {
						cutpoints_by_var[dep_idx].insert(v);
					}
				}
			}
		}

		let mut partitions_by_numeric_var: Vec<Vec<Interval>> = Vec::with_capacity(num_numeric_vars);
		let mut numeric_domain_sizes: Vec<usize> = Vec::with_capacity(num_numeric_vars);
		for (var_id, var) in task.numeric_variables().iter().enumerate() {
			let parts = match var.get_type() {
				NumericType::Constant => vec![Interval::singleton(initial_numeric_values[var_id])],
				NumericType::Regular => {
					let cuts: Vec<f64> = cutpoints_by_var[var_id].iter().map(|v| v.into_inner()).collect();
					if cuts.is_empty() {
						vec![Interval::unbounded()]
					} else {
						partitions_from_cutpoints(&cuts)
					}
				}
				NumericType::Derived | NumericType::Cost => vec![Interval::unbounded()],
			};
			numeric_domain_sizes.push(parts.len());
			partitions_by_numeric_var.push(parts);
		}

		let partitions = NumericPartitions::with_partitions(partitions_by_numeric_var);
		Self {
			partitions,
			numeric_domain_sizes,
			comparison_index,
			comparison_trees,
		}
	}

	pub fn partitions(&self) -> &NumericPartitions {
		&self.partitions
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
	) -> AbstractOperatorGenerator {
		AbstractOperatorGenerator::new_with_identity_mapping(
			task,
			self.partitions.clone(),
			self.numeric_domain_sizes.clone(),
			combine_labels,
		)
	}

	/// Builds the *reachable* abstract state space starting from the initial abstract state.
	///
	/// This mirrors numeric-fd's factory behavior (but avoids allocating `num_states` upfront):
	/// - States include propositional variables (incl. comparison-axiom vars) and numeric partitions.
	/// - Comparison variables are (re-)evaluated from numeric partition intervals; if undecidable,
	///   both truth values are enumerated.
	pub fn build_abstract_state_space(
		&self,
		task: &dyn AbstractNumericTask,
		combine_labels: bool,
	) -> AbstractStateSpace {
		let mut generator = self.make_operator_generator(task, combine_labels);
		let operators = generator.build_abstract_operators(task);
		self.build_state_space_with_operators(task, &generator, &operators)
	}

	/// Computes an abstract wildcard plan (sequence of per-step concrete-op-ID sets) by:
	/// 1) building the reachable abstract state space,
	/// 2) running reverse Dijkstra from abstract goals,
	/// 3) extracting a decreasing-distance path and collecting all cheapest realizations per step.
	pub fn compute_wildcard_plan(
		&self,
		task: &dyn AbstractNumericTask,
		combine_labels: bool,
	) -> Option<WildcardPlanResult> {
		let mut generator = self.make_operator_generator(task, combine_labels);
		let operators = generator.build_abstract_operators(task);
		let space = self.build_state_space_with_operators(task, &generator, &operators);
		compute_wildcard_plan_from_space(task, &generator, &operators, &space)
	}

	fn build_state_space_with_operators(
		&self,
		task: &dyn AbstractNumericTask,
		generator: &AbstractOperatorGenerator,
		operators: &[AbstractOperator],
	) -> AbstractStateSpace {
		let domain_sizes = generator.domain_sizes();
		let domain_mapping = generator.domain_mapping();
		let hash_multipliers = generator.hash_multipliers();
		let num_props = domain_sizes.len();
		let numeric_domain_sizes = generator.numeric_domain_sizes();
		let comparison_var_ids: Vec<usize> = self
			.comparison_trees
			.iter()
			.filter_map(|t| usize::try_from(t.affected_var_id).ok())
			.collect();

		let goal_facts = compute_abstract_goals(task, domain_mapping);

		let base_init = compute_initial_base_state_hash(
			task,
			domain_mapping,
			domain_sizes,
			numeric_domain_sizes,
			hash_multipliers,
			&self.partitions,
			&comparison_var_ids,
		);
		let init_hashes = enumerate_states_with_evaluated_comparisons(
			base_init,
			task,
			domain_sizes,
			domain_mapping,
			hash_multipliers,
			numeric_domain_sizes,
			&self.partitions,
			&self.comparison_trees,
			&comparison_var_ids,
			&[],
		);

		let num_states = compute_num_states(domain_sizes, numeric_domain_sizes);
		let mut goal_hashes: Vec<i32> = Vec::new();
		for s in 0..num_states {
			let h = s as i32;
			if !is_goal_state(h, &goal_facts, num_props, domain_sizes, numeric_domain_sizes, hash_multipliers)
			{
				continue;
			}
			let alts = enumerate_states_with_evaluated_comparisons(
				h,
				task,
				domain_sizes,
				domain_mapping,
				hash_multipliers,
				numeric_domain_sizes,
				&self.partitions,
				&self.comparison_trees,
				&comparison_var_ids,
				&[],
			);
			if alts.binary_search(&h).is_ok() {
				goal_hashes.push(h);
			}
		}

		let match_tree = MatchTree::build(
			domain_sizes,
			numeric_domain_sizes,
			hash_multipliers,
			operators,
			&comparison_var_ids,
		);

		let mut states: Vec<i32> = Vec::new();
		let mut index_by_hash: HashMap<i32, usize> = HashMap::new();
		let mut outgoing: Vec<Vec<AbstractTransition>> = Vec::new();
		let mut incoming: Vec<Vec<(usize, usize, f64)>> = Vec::new();
		let mut queue: VecDeque<usize> = VecDeque::new();

		// Seed with goal states (regression) like numeric-fd does.
		for h in goal_hashes {
			if index_by_hash.contains_key(&h) {
				continue;
			}
			let id = states.len();
			states.push(h);
			index_by_hash.insert(h, id);
			outgoing.push(Vec::new());
			incoming.push(Vec::new());
			queue.push_back(id);
		}

		// Also keep initial states around for plan extraction, but don't expand them.
		let mut initial_state_ids: Vec<usize> = Vec::new();
		for h in init_hashes {
			let id = if let Some(&id) = index_by_hash.get(&h) {
				id
			} else {
				let id = states.len();
				states.push(h);
				index_by_hash.insert(h, id);
				outgoing.push(Vec::new());
				incoming.push(Vec::new());
				id
			};
			initial_state_ids.push(id);
		}
		initial_state_ids.sort_unstable();
		initial_state_ids.dedup();

		let mut applicable_operator_ids: Vec<usize> = Vec::new();
		while let Some(succ_id) = queue.pop_front() {
			let succ_hash = states[succ_id];
			let succ_base = reset_comparison_vars_to_unknown_except(
				succ_hash,
				domain_sizes,
				domain_mapping,
				hash_multipliers,
				&comparison_var_ids,
				&[],
			);

			match_tree.get_applicable_operator_ids(succ_base, &mut applicable_operator_ids);
			for &op_id in &applicable_operator_ids {
				let op = &operators[op_id];
				let pred_base = succ_base.wrapping_add(op.hash_effect);
				let fixed_comparisons = get_comparison_preconditions(op, &comparison_var_ids);
				let predecessors = enumerate_states_with_evaluated_comparisons(
					pred_base,
					task,
					domain_sizes,
					domain_mapping,
					hash_multipliers,
					numeric_domain_sizes,
					&self.partitions,
					&self.comparison_trees,
					&comparison_var_ids,
					&fixed_comparisons,
				);

				for pred_hash in predecessors {
					let pred_id = if let Some(&id) = index_by_hash.get(&pred_hash) {
						id
					} else {
						let id = states.len();
						states.push(pred_hash);
						index_by_hash.insert(pred_hash, id);
						outgoing.push(Vec::new());
						incoming.push(Vec::new());
						queue.push_back(id);
						id
					};

					outgoing[pred_id].push(AbstractTransition {
						op_id,
						successor: succ_id,
						cost: op.cost,
					});
					incoming[succ_id].push((pred_id, op_id, op.cost));
				}
			}
		}

		AbstractStateSpace {
			states,
			outgoing,
			incoming,
			initial_state_ids,
			goal_facts,
		}
	}
}

fn compute_num_states(domain_sizes: &[i32], numeric_domain_sizes: &[usize]) -> i32 {
	let mut num: i64 = 1;
	for &s in domain_sizes {
		num = num.saturating_mul(i64::from(s.max(0)));
	}
	for &s in numeric_domain_sizes {
		num = num.saturating_mul(i64::try_from(s).unwrap_or(0));
	}
	if num <= 0 {
		return 0;
	}
	if num > i64::from(i32::MAX) {
		// If this overflows the hashing scheme, don't attempt a full scan.
		return 0;
	}
	num as i32
}

fn get_comparison_preconditions(op: &AbstractOperator, comparison_var_ids: &[usize]) -> Vec<planners_sas::numeric::numeric_task::Fact> {
	let mut out: Vec<planners_sas::numeric::numeric_task::Fact> = Vec::new();
	for f in &op.preconditions {
		let Ok(var) = usize::try_from(f.var()) else {
			continue;
		};
		if comparison_var_ids.contains(&var) {
			out.push(f.clone());
		}
	}
	out
}

fn compute_wildcard_plan_from_space(
	task: &dyn AbstractNumericTask,
	generator: &AbstractOperatorGenerator,
	operators: &[AbstractOperator],
	space: &AbstractStateSpace,
) -> Option<WildcardPlanResult> {
	let domain_sizes = generator.domain_sizes();
	let domain_mapping = generator.domain_mapping();
	let hash_multipliers = generator.hash_multipliers();
	let num_props = domain_sizes.len();
	let numeric_domain_sizes = generator.numeric_domain_sizes();

	let mut goal_state_ids: Vec<usize> = Vec::new();
	for (id, &hash) in space.states.iter().enumerate() {
		if is_goal_state(hash, &space.goal_facts, num_props, domain_sizes, numeric_domain_sizes, hash_multipliers)
		{
			goal_state_ids.push(id);
		}
	}
	if goal_state_ids.is_empty() {
		return None;
	}

	let (dist, generating_op, next_state) = reverse_dijkstra(space, &goal_state_ids);

	let mut best_init: Option<usize> = None;
	let mut best_d: f64 = f64::INFINITY;
	for &init_id in &space.initial_state_ids {
		let d = dist[init_id];
		if d.is_finite() && d < best_d {
			best_d = d;
			best_init = Some(init_id);
		}
	}
	let mut current = best_init?;

	let _ = task;
	let _ = domain_mapping;

	let mut wildcard_plan: Vec<Vec<usize>> = Vec::new();
	let mut abstract_state_ids: Vec<usize> = vec![current];

	// For debugging / parity with numeric-fd deviation code.
	let mut abstract_prop_states: Vec<Vec<i32>> = Vec::new();
	let mut abstract_numeric_states: Vec<Vec<i32>> = Vec::new();
	decode_state_to_vectors(
		space.states[current],
		num_props,
		domain_sizes,
		numeric_domain_sizes,
		hash_multipliers,
		&mut abstract_prop_states,
		&mut abstract_numeric_states,
	);

	let mut safety_steps = 0usize;
	while !is_goal_state(
		space.states[current],
		&space.goal_facts,
		num_props,
		domain_sizes,
		numeric_domain_sizes,
		hash_multipliers,
	) {
		safety_steps += 1;
		if safety_steps > space.states.len() + 1 {
			return None;
		}
		let op_id = generating_op.get(current).copied().flatten()?;
		let succ_id = next_state.get(current).copied().flatten()?;
		let step_cost = operators[op_id].cost;

		let mut concrete_ids: HashSet<usize> = HashSet::new();
		for (cand_op_id, cand_op) in operators.iter().enumerate() {
			if (cand_op.cost - step_cost).abs() > 1e-9 {
				continue;
			}
			if !operator_is_applicable(
				space.states[current],
				cand_op,
				num_props,
				domain_sizes,
				numeric_domain_sizes,
				hash_multipliers,
			) {
				continue;
			}

			// In the built state space we already have the enumerated successors.
			let mut reaches_succ = false;
			for edge in &space.outgoing[current] {
				if edge.op_id == cand_op_id && edge.successor == succ_id {
					reaches_succ = true;
					break;
				}
			}
			if !reaches_succ {
				continue;
			}

			for &cid in &cand_op.concrete_op_ids {
				concrete_ids.insert(cid);
			}
		}
		let mut step: Vec<usize> = concrete_ids.into_iter().collect();
		step.sort_unstable();
		wildcard_plan.push(step);

		current = succ_id;
		abstract_state_ids.push(current);
		decode_state_to_vectors(
			space.states[current],
			num_props,
			domain_sizes,
			numeric_domain_sizes,
			hash_multipliers,
			&mut abstract_prop_states,
			&mut abstract_numeric_states,
		);
	}

	Some(WildcardPlanResult {
		wildcard_plan,
		abstract_state_ids,
		abstract_prop_states,
		abstract_numeric_states,
	})
}

fn reverse_dijkstra(
	space: &AbstractStateSpace,
	goal_state_ids: &[usize],
) -> (Vec<f64>, Vec<Option<usize>>, Vec<Option<usize>>) {
	let n = space.states.len();
	let mut dist: Vec<f64> = vec![f64::INFINITY; n];
	let mut generating_op: Vec<Option<usize>> = vec![None; n];
	let mut next_state: Vec<Option<usize>> = vec![None; n];

	let mut heap: BinaryHeap<(Reverse<NotNan<f64>>, usize)> = BinaryHeap::new();
	for &g in goal_state_ids {
		dist[g] = 0.0;
		heap.push((Reverse(NotNan::new(0.0).unwrap()), g));
	}

	while let Some((Reverse(d), state)) = heap.pop() {
		let d = d.into_inner();
		if d > dist[state] + 1e-12 {
			continue;
		}
		for &(pred, op_id, cost) in &space.incoming[state] {
			let alt = d + cost;
			if alt + 1e-12 < dist[pred] {
				dist[pred] = alt;
				generating_op[pred] = Some(op_id);
				next_state[pred] = Some(state);
				heap.push((Reverse(NotNan::new(alt).unwrap()), pred));
			}
		}
	}

	(dist, generating_op, next_state)
}

fn operator_is_applicable(
	state_hash: i32,
	op: &AbstractOperator,
	num_props: usize,
	domain_sizes: &[i32],
	numeric_domain_sizes: &[usize],
	hash_multipliers: &[i32],
) -> bool {
	for pre in &op.preconditions {
		let var = pre.var() as usize;
		let expected = pre.value();
		let mult = hash_multipliers[var] as i64;
		let state = state_hash as i64;
		let dom_size: i64 = if var < num_props {
			domain_sizes[var] as i64
		} else {
			let n = var - num_props;
			numeric_domain_sizes.get(n).copied().unwrap_or(0) as i64
		};
		if dom_size <= 0 {
			return false;
		}
		let actual = ((state / mult) % dom_size) as i32;
		if actual != expected {
			return false;
		}
	}
	true
}

fn compute_abstract_goals(
	task: &dyn AbstractNumericTask,
	domain_mapping: &DomainMapping,
) -> Vec<planners_sas::numeric::numeric_task::Fact> {
	use planners_sas::numeric::numeric_task::Fact;

	let mut goal_axiom_map: HashMap<u32, usize> = HashMap::new();
	for (idx, ax) in task.axioms().iter().enumerate() {
		if !ax.conditions().is_empty() {
			goal_axiom_map.insert(ax.var_id(), idx);
		}
	}

	let mut out: Vec<Fact> = Vec::new();
	for i in 0..task.get_num_goals() {
		let g = task.get_goal_fact(i);
		let var = g.var() as u32;
		if let Some(&ax_idx) = goal_axiom_map.get(&var) {
			let ax = &task.axioms()[ax_idx];
			for cond in ax.conditions() {
				let v = cond.var() as usize;
				let val = cond.value() as usize;
				let mapped = domain_mapping
					.get(v)
					.and_then(|m| m.get(val))
					.copied()
					.unwrap_or(cond.value());
				out.push(Fact::new(cond.var() as u32, mapped));
			}
		} else {
			let v = g.var() as usize;
			let val = g.value() as usize;
			let mapped = domain_mapping
				.get(v)
				.and_then(|m| m.get(val))
				.copied()
				.unwrap_or(g.value());
			out.push(Fact::new(g.var() as u32, mapped));
		}
	}

	out.sort();
	out.dedup();
	out
}

fn is_goal_state(
	state_hash: i32,
	goals: &[planners_sas::numeric::numeric_task::Fact],
	num_props: usize,
	domain_sizes: &[i32],
	numeric_domain_sizes: &[usize],
	hash_multipliers: &[i32],
) -> bool {
	for g in goals {
		let var = g.var() as usize;
		let expected = g.value();
		let mult = hash_multipliers[var] as i64;
		let state = state_hash as i64;
		let dom_size: i64 = if var < num_props {
			domain_sizes[var] as i64
		} else {
			let n = var - num_props;
			numeric_domain_sizes.get(n).copied().unwrap_or(0) as i64
		};
		if dom_size <= 0 {
			return false;
		}
		let actual = ((state / mult) % dom_size) as i32;
		if actual != expected {
			return false;
		}
	}
	true
}

fn compute_initial_base_state_hash(
	task: &dyn AbstractNumericTask,
	domain_mapping: &DomainMapping,
	domain_sizes: &[i32],
	numeric_domain_sizes: &[usize],
	hash_multipliers: &[i32],
	partitions: &NumericPartitions,
	comparison_var_ids: &[usize],
) -> i32 {
	let num_props = domain_sizes.len();
	let mut is_comparison: Vec<bool> = vec![false; num_props];
	for &v in comparison_var_ids {
		if v < is_comparison.len() {
			is_comparison[v] = true;
		}
	}

	let prop_init = task.get_initial_propositional_state_values();
	let num_init = task.get_initial_numeric_state_values();

	let mut index: i64 = 0;
	for var in 0..num_props {
		let mult = hash_multipliers[var] as i64;
		let abs_val: i32 = if is_comparison[var] {
			// Start from UNKNOWN like numeric-fd does before enumeration.
			domain_mapping[var]
				.get(COMPARISON_UNKNOWN_VAL as usize)
				.copied()
				.unwrap_or(COMPARISON_UNKNOWN_VAL)
		} else {
			let concrete_val = prop_init[var];
			let cidx = usize::try_from(concrete_val).unwrap_or(0);
			domain_mapping[var].get(cidx).copied().unwrap_or(concrete_val)
		};
		index += mult * (abs_val as i64);
	}

	for num_var_id in 0..numeric_domain_sizes.len() {
		let abs_var = num_props + num_var_id;
		let mult = hash_multipliers[abs_var] as i64;
		let val = *num_init.get(num_var_id).unwrap_or(&0.0);
		let part = partitions
			.partitions(num_var_id)
			.and_then(|parts| partition_for_value(parts, val))
			.unwrap_or(0);
		index += mult * (part as i64);
	}

	index as i32
}

fn enumerate_states_with_evaluated_comparisons(
	base_state_hash: i32,
	task: &dyn AbstractNumericTask,
	domain_sizes: &[i32],
	domain_mapping: &DomainMapping,
	hash_multipliers: &[i32],
	numeric_domain_sizes: &[usize],
	partitions: &NumericPartitions,
	comparison_trees: &[ComparisonTree],
	comparison_var_ids: &[usize],
	fixed_comparisons: &[planners_sas::numeric::numeric_task::Fact],
) -> Vec<i32> {
	let num_props = domain_sizes.len();
	let state_unknown = reset_comparison_vars_to_unknown_except(
		base_state_hash,
		domain_sizes,
		domain_mapping,
		hash_multipliers,
		comparison_var_ids,
		fixed_comparisons,
	);

	let numeric_intervals = build_numeric_intervals(
		base_state_hash,
		num_props,
		numeric_domain_sizes,
		hash_multipliers,
		partitions,
		task,
	);

	let mut fixed_map: HashMap<u32, i32> = HashMap::new();
	for f in fixed_comparisons {
		fixed_map.insert(f.var(), f.value());
	}

	let mut states: Vec<i32> = vec![state_unknown];
	for tree in comparison_trees {
		let Ok(var_id) = usize::try_from(tree.affected_var_id) else {
			continue;
		};
		if var_id >= num_props {
			continue;
		}
		if domain_sizes[var_id] <= 1 {
			continue;
		}
		if fixed_map.contains_key(&(var_id as u32)) {
			// If the fixed value contradicts a definite evaluation, no states are feasible.
			if let Some(eval) = tree.evaluate_interval(&numeric_intervals) {
				let required = if eval { COMPARISON_TRUE_VAL } else { COMPARISON_FALSE_VAL };
				if fixed_map[&(var_id as u32)] != required {
					return Vec::new();
				}
			}
			continue;
		}

		let mult = hash_multipliers[var_id];
		let unknown_abs = domain_mapping[var_id]
			.get(COMPARISON_UNKNOWN_VAL as usize)
			.copied()
			.unwrap_or(COMPARISON_UNKNOWN_VAL);
		let delta_true = (domain_mapping[var_id]
			.get(COMPARISON_TRUE_VAL as usize)
			.copied()
			.unwrap_or(COMPARISON_TRUE_VAL)
			- unknown_abs)
			* mult;
		let delta_false = (domain_mapping[var_id]
			.get(COMPARISON_FALSE_VAL as usize)
			.copied()
			.unwrap_or(COMPARISON_FALSE_VAL)
			- unknown_abs)
			* mult;

		match tree.evaluate_interval(&numeric_intervals) {
			Some(true) => {
				for s in &mut states {
					*s += delta_true;
				}
			}
			Some(false) => {
				for s in &mut states {
					*s += delta_false;
				}
			}
			None => {
				let mut next: Vec<i32> = Vec::with_capacity(states.len() * 2);
				for &s in &states {
					next.push(s + delta_true);
					next.push(s + delta_false);
				}
				states = next;
			}
		}
	}

	states.sort_unstable();
	states.dedup();
	states
}

fn reset_comparison_vars_to_unknown_except(
	state_hash: i32,
	domain_sizes: &[i32],
	domain_mapping: &DomainMapping,
	hash_multipliers: &[i32],
	comparison_var_ids: &[usize],
	fixed_comparisons: &[planners_sas::numeric::numeric_task::Fact],
) -> i32 {
	let mut fixed: HashSet<u32> = HashSet::new();
	for f in fixed_comparisons {
		fixed.insert(f.var());
	}

	let mut out = state_hash;
	for &var_id in comparison_var_ids {
		if var_id >= domain_sizes.len() {
			continue;
		}
		if fixed.contains(&(var_id as u32)) {
			continue;
		}
		if domain_sizes[var_id] <= 1 {
			continue;
		}
		let mult = hash_multipliers[var_id] as i64;
		let dom = domain_sizes[var_id] as i64;
		let cur = (((out as i64) / mult) % dom) as i32;
		let unknown_abs = domain_mapping[var_id]
			.get(COMPARISON_UNKNOWN_VAL as usize)
			.copied()
			.unwrap_or(COMPARISON_UNKNOWN_VAL);
		out += ((unknown_abs - cur) as i64 * mult) as i32;
	}
	out
}

fn build_numeric_intervals(
	state_hash: i32,
	num_props: usize,
	numeric_domain_sizes: &[usize],
	hash_multipliers: &[i32],
	partitions: &NumericPartitions,
	task: &dyn AbstractNumericTask,
) -> Vec<Interval> {
	let num_numeric_vars = task.numeric_variables().len();
	let initial_numeric_values = task.get_initial_numeric_state_values();
	let mut out: Vec<Interval> = vec![Interval::unbounded(); num_numeric_vars];
	for (i, v) in task.numeric_variables().iter().enumerate() {
		if v.get_type() == &NumericType::Constant {
			out[i] = Interval::singleton(initial_numeric_values[i]);
			continue;
		}
		if i >= numeric_domain_sizes.len() {
			continue;
		}
		let abs_var = num_props + i;
		let mult = hash_multipliers[abs_var] as i64;
		let dom = numeric_domain_sizes[i] as i64;
		let part = (((state_hash as i64) / mult) % dom) as usize;
		if let Some(iv) = partitions.partition_interval(i, part) {
			out[i] = iv;
		}
	}
	out
}

fn partition_for_value(partitions: &[Interval], value: f64) -> Option<usize> {
	for (i, &iv) in partitions.iter().enumerate() {
		if interval_contains(iv, value) {
			return Some(i);
		}
	}
	None
}

fn interval_contains(iv: Interval, x: f64) -> bool {
	if iv.is_empty() {
		return false;
	}
	let lower_ok = if x > iv.lower {
		true
	} else if x == iv.lower {
		iv.lower_closed
	} else {
		false
	};
	let upper_ok = if x < iv.upper {
		true
	} else if x == iv.upper {
		iv.upper_closed
	} else {
		false
	};
	lower_ok && upper_ok
}

fn decode_state_to_vectors(
	state_hash: i32,
	num_props: usize,
	domain_sizes: &[i32],
	numeric_domain_sizes: &[usize],
	hash_multipliers: &[i32],
	prop_out: &mut Vec<Vec<i32>>,
	num_out: &mut Vec<Vec<i32>>,
) {
	let mut props: Vec<i32> = Vec::with_capacity(num_props);
	for var_id in 0..num_props {
		let mult = hash_multipliers[var_id] as i64;
		let dom = domain_sizes[var_id] as i64;
		let val = (((state_hash as i64) / mult) % dom) as i32;
		props.push(val);
	}
	let mut nums: Vec<i32> = Vec::with_capacity(numeric_domain_sizes.len());
	for (num_id, &dom_u) in numeric_domain_sizes.iter().enumerate() {
		let abs_var = num_props + num_id;
		let mult = hash_multipliers[abs_var] as i64;
		let dom = dom_u as i64;
		let part = (((state_hash as i64) / mult) % dom) as i32;
		nums.push(part);
	}
	prop_out.push(props);
	num_out.push(nums);
}

fn constant_leaf_values(
	tree: &ComparisonTree,
	task: &dyn AbstractNumericTask,
	initial_numeric_values: &[f64],
) -> Vec<f64> {
	let num_numeric_vars = task.numeric_variables().len();
	let mut out: HashSet<u64> = HashSet::new();
	for node in &tree.nodes {
		let super::comparison_expression::ComparisonTreeNode::Leaf { numeric_var_id } = node else {
			continue;
		};
		let Ok(idx) = usize::try_from(*numeric_var_id) else {
			continue;
		};
		if idx >= num_numeric_vars {
			continue;
		}
		if task.numeric_variables()[idx].get_type() != &NumericType::Constant {
			continue;
		}
		let v = initial_numeric_values[idx];
		if v.is_nan() {
			continue;
		}
		out.insert(v.to_bits());
	}
	out.into_iter().map(f64::from_bits).collect()
}

fn partitions_from_cutpoints(cutpoints: &[f64]) -> Vec<Interval> {
	let mut cuts: Vec<f64> = cutpoints
		.iter()
		.copied()
		.filter(|v| v.is_finite() && !v.is_nan())
		.collect();
	cuts.sort_by(|a, b| a.partial_cmp(b).unwrap());
	cuts.dedup_by(|a, b| a.to_bits() == b.to_bits());

	let mut out: Vec<Interval> = Vec::new();
	let mut prev = f64::NEG_INFINITY;
	for &c in &cuts {
		out.push(Interval::new(prev, c, false, false));
		out.push(Interval::singleton(c));
		prev = c;
	}
	out.push(Interval::new(prev, f64::INFINITY, false, false));
	out
}

#[cfg(test)]
mod tests {
	use super::*;

	use planners_sas::numeric::axioms::{ComparisonAxiom, ComparisonOperator};
	use planners_sas::numeric::numeric_task::{
		ExplicitVariable, Fact, Metric, NumericRootTask, NumericVariable, Operator,
	};

	#[test]
	fn factory_splits_regular_var_at_constants_in_comparison_trees() {
		let variables = vec![ExplicitVariable::new(
			3,
			"cmp".into(),
			vec!["true".into(), "false".into(), "unknown".into()],
			0,
			2,
		)];

		let numeric_variables = vec![
			NumericVariable::new("x0".into(), NumericType::Regular, -1),
			NumericVariable::new("c10".into(), NumericType::Constant, -1),
		];

		let comparison_axioms = vec![ComparisonAxiom::new(
			0,
			0,
			1,
			ComparisonOperator::LessThan,
		)];

		let op = Operator::new("op".into(), vec![Fact::new(0, 0)], vec![], vec![], 1);

		let task = NumericRootTask::new(
			4,
			Metric::new(true, -1),
			variables,
			numeric_variables,
			vec![],
			vec![],
			vec![0],
			vec![0.0, 10.0],
			vec![op],
			vec![],
			comparison_axioms,
			vec![],
			(0, 0),
		);

		let factory = DomainAbstractionFactory::from_task(&task);
		assert_eq!(factory.numeric_domain_sizes(), &[3, 1]);

		let x0_parts = factory.partitions().partitions(0).unwrap();
		assert_eq!(x0_parts.len(), 3);
		assert_eq!(x0_parts[0], Interval::new(f64::NEG_INFINITY, 10.0, false, false));
		assert_eq!(x0_parts[1], Interval::singleton(10.0));
		assert_eq!(x0_parts[2], Interval::new(10.0, f64::INFINITY, false, false));

		let c10_parts = factory.partitions().partitions(1).unwrap();
		assert_eq!(c10_parts, &[Interval::singleton(10.0)]);

		// Smoke-test that generator can be created.
		let _gen = factory.make_operator_generator(&task, false);
	}

	#[test]
	fn enumerate_states_branches_on_undecidable_comparison() {
		use planners_sas::numeric::numeric_task::{ExplicitVariable, Fact, Metric, NumericRootTask, NumericType, NumericVariable, Operator};
		use planners_sas::numeric::axioms::{ComparisonAxiom, ComparisonOperator};

		let variables = vec![ExplicitVariable::new(
			3,
			"cmp".into(),
			vec!["true".into(), "false".into(), "unknown".into()],
			0,
			2,
		)];
		let numeric_variables = vec![
			NumericVariable::new("x".into(), NumericType::Regular, -1),
			NumericVariable::new("y".into(), NumericType::Regular, -1),
		];
		let comparison_axioms = vec![ComparisonAxiom::new(0, 0, 1, ComparisonOperator::LessThan)];
		let op = Operator::new("noop".into(), vec![], vec![], vec![], 1);
		let task = NumericRootTask::new(
			4,
			Metric::new(true, -1),
			variables,
			numeric_variables,
			vec![],
			vec![],
			vec![COMPARISON_UNKNOWN_VAL],
			vec![0.0, 0.0],
			vec![op],
			vec![],
			comparison_axioms,
			vec![],
			(0, 0),
		);

		let factory = DomainAbstractionFactory::from_task(&task);
		let mut generator = factory.make_operator_generator(&task, true);
		let hash_multipliers = generator.hash_multipliers().to_vec();
		let domain_sizes = generator.domain_sizes().to_vec();
		let domain_mapping = generator.domain_mapping().clone();
		let base = compute_initial_base_state_hash(
			&task,
			&domain_mapping,
			&domain_sizes,
			generator.numeric_domain_sizes(),
			&hash_multipliers,
			factory.partitions(),
			&[0],
		);
		let states = enumerate_states_with_evaluated_comparisons(
			base,
			&task,
			&domain_sizes,
			&domain_mapping,
			&hash_multipliers,
			generator.numeric_domain_sizes(),
			factory.partitions(),
			factory.comparison_trees(),
			&[0],
			&[],
		);
		assert_eq!(states.len(), 2);
	}

	#[test]
	fn wildcard_plan_collects_all_equivalent_concrete_ops() {
		use planners_sas::numeric::numeric_task::{ExplicitVariable, Fact, Metric, NumericRootTask, NumericType, NumericVariable, Operator};

		let variables = vec![ExplicitVariable::new(
			2,
			"v".into(),
			vec!["v0".into(), "v1".into()],
			0,
			0,
		)];
		let numeric_variables: Vec<NumericVariable> = vec![];
		let goals = vec![Fact::new(0, 1)];
		let op0 = Operator::new("set0".into(), vec![Fact::new(0, 0)], vec![planners_sas::numeric::numeric_task::Effect::new(vec![], 0, 0, 1)], vec![], 1);
		let op1 = Operator::new("set1".into(), vec![Fact::new(0, 0)], vec![planners_sas::numeric::numeric_task::Effect::new(vec![], 0, 0, 1)], vec![], 1);
		let task = NumericRootTask::new(
			4,
			Metric::new(true, -1),
			variables,
			numeric_variables,
			goals,
			vec![],
			vec![0],
			vec![],
			vec![op0, op1],
			vec![],
			vec![],
			vec![],
			(0, 0),
		);

		let factory = DomainAbstractionFactory::from_task(&task);
		let result = factory.compute_wildcard_plan(&task, true).expect("plan exists");
		assert_eq!(result.wildcard_plan.len(), 1);
		assert_eq!(result.wildcard_plan[0], vec![0, 1]);
	}
}
