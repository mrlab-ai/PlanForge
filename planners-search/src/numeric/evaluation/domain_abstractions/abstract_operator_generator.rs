#[cfg(test)]
mod tests;

use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};

use anyhow::{Context, Result, anyhow, ensure};

use planners_sas::numeric::axioms::{CalOperator, ComparisonOperator};

use planners_sas::numeric::numeric_task::{
    AbstractNumericTask, AssignmentEffect, Effect, Fact, NumericType, Operator,
    metric_operator_cost_from_initial_values,
};

use super::comparison_expression::{ArithOp, ComparisonTree, Interval};
use super::domain_abstraction::{ComparisonAxiomIndex, NumericPartitions};
use super::numeric_context::propagate_assignment_axiom_intervals;
use super::utils;

const COMPARISON_TRUE_VAL: usize = 0;
const COMPARISON_FALSE_VAL: usize = 1;
const COMPARISON_UNKNOWN_VAL: usize = 2;

pub type DomainMapping = Vec<Vec<i32>>;

#[derive(Debug, Clone, PartialEq)]
pub struct AbstractOperator {
    pub concrete_op_ids: Vec<usize>,
    pub cost: f64,
    pub hash_effect: i32,
    pub regression_preconditions: Vec<Fact>,
    pub preconditions: Vec<Fact>,
    pub changed_numeric_vars: Vec<usize>,
}

impl AbstractOperator {
    pub fn new(
        prev_pairs: &[Fact],
        pre_pairs: &[Fact],
        eff_pairs: &[Fact],
        cost: f64,
        hash_multipliers: &[i32],
        concrete_op_ids: Vec<usize>,
        changed_numeric_vars: Vec<usize>,
    ) -> Self {
        let mut preconditions: Vec<Fact> = pre_pairs.to_vec();
        preconditions.extend_from_slice(prev_pairs);
        preconditions.sort();
        debug_assert!(preconditions.windows(2).all(|w| w[0].var() != w[1].var()));
        debug_assert!(preconditions.windows(2).all(|w| w[0].var() != w[1].var()));

        let mut regression_preconditions: Vec<Fact> = prev_pairs.to_vec();
        regression_preconditions.extend_from_slice(eff_pairs);
        regression_preconditions.sort();
        debug_assert!(
            regression_preconditions
                .windows(2)
                .all(|w| w[0].var() != w[1].var())
        );

        debug_assert_eq!(
            pre_pairs.len(),
            eff_pairs.len(),
            "abstract operator pre/eff pair mismatch: pre_pairs={pre_pairs:?} eff_pairs={eff_pairs:?}"
        );

        let mut hash_effect: i32 = 0;
        for (pre, eff) in pre_pairs.iter().zip(eff_pairs.iter()) {
            debug_assert_eq!(
                pre.var(),
                eff.var(),
                "abstract operator transition var mismatch: pre={pre:?} eff={eff:?}"
            );

            let var = pre.var() as usize;
            let multiplier = hash_multipliers[var];
            let new_val = pre.value();
            let old_val = eff.value();
            hash_effect += (new_val - old_val) * multiplier;
        }

        Self {
            concrete_op_ids,
            cost,
            hash_effect,
            regression_preconditions,
            preconditions,
            changed_numeric_vars,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TransitionInfo {
    pub source_partition_facts: Vec<Fact>,
    pub target_partition_facts: Vec<Fact>,
    pub prevail_facts: Vec<Fact>,
    pub changed_numeric_vars: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct OperatorSignature {
    prev_pairs: Vec<(u32, i32)>,
    pre_pairs: Vec<(u32, i32)>,
    eff_pairs: Vec<(u32, i32)>,
    cost_bits: u64,
}

fn arith_op_from_axiom(operator: &CalOperator) -> ArithOp {
    match operator {
        CalOperator::Sum => ArithOp::Add,
        CalOperator::Difference => ArithOp::Sub,
        CalOperator::Product => ArithOp::Mul,
        CalOperator::Division => ArithOp::Div,
    }
}

#[derive(Clone)]
pub struct AbstractOperatorGenerator {
    domain_mapping: DomainMapping,
    domain_sizes: Vec<i32>,
    numeric_domain_sizes: Vec<usize>,
    hash_multipliers: Vec<i32>,
    partitions: NumericPartitions,
    comparison_index: Option<ComparisonAxiomIndex>,
    comparison_trees: Vec<ComparisonTree>,
    comparisons_by_numeric_dep: Vec<Vec<usize>>,
    derived_prop_vars: HashSet<u32>,
    combine_labels: bool,
}

impl AbstractOperatorGenerator {
    pub fn new(
        task: &dyn AbstractNumericTask,
        domain_mapping: DomainMapping,
        domain_sizes: Vec<i32>,
        partitions: NumericPartitions,
        numeric_domain_sizes: Vec<usize>,
        combine_labels: bool,
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

            let var_i32 = i32::try_from(var).context("var index does not fit i32")?;
            let concrete_size = task
                .get_variable_domain_size(var_i32)
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
                domain_mapping[var].len() == concrete_size as usize,
                "domain_mapping[{var}] has len {}, expected concrete size {concrete_size}",
                domain_mapping[var].len()
            );
            for (val, &mapped) in domain_mapping[var].iter().enumerate() {
                ensure!(
                    mapped >= 0 && mapped < abs_size,
                    "domain_mapping[{var}][{val}]={mapped} out of range for abstract size {abs_size}"
                );
            }
        }
        for (n, &parts) in numeric_domain_sizes.iter().enumerate() {
            ensure!(parts > 0, "numeric_domain_sizes[{n}] must be > 0");
        }

        let hash_multipliers = compute_hash_multipliers(&domain_sizes, &numeric_domain_sizes)?;

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

        let mut comparisons_by_numeric_dep: Vec<Vec<usize>> =
            vec![Vec::new(); task.numeric_variables().len()];
        for (tree_idx, tree) in comparison_trees.iter().enumerate() {
            for dep in tree.regular_numeric_var_dependencies(task) {
                let dep_idx = usize::try_from(dep).map_err(|_| {
                    anyhow!("regular_numeric_var_dependencies returned non-usize index: {dep}")
                })?;
                ensure!(
                    dep_idx < comparisons_by_numeric_dep.len(),
                    "comparison tree depends on numeric var {dep_idx}, but only {} numeric vars exist",
                    comparisons_by_numeric_dep.len()
                );
                comparisons_by_numeric_dep[dep_idx].push(tree_idx);
            }
        }

        let derived_prop_vars: HashSet<u32> = task
            .comparison_axioms()
            .iter()
            .map(|ax| ax.get_affected_var_id() as u32)
            .collect();

        Ok(Self {
            domain_mapping,
            domain_sizes,
            numeric_domain_sizes,
            hash_multipliers,
            partitions,
            comparison_index,
            comparison_trees,
            comparisons_by_numeric_dep,
            derived_prop_vars,
            combine_labels,
        })
    }

    /// Convenience constructor that mirrors numeric-fd's default setup when no CEGAR mapping
    /// exists yet: identity mapping for non-derived variables, and a 3-valued mapping
    /// (false/true/unknown) for comparison-axiom variables.
    pub fn new_with_identity_mapping(
        task: &dyn AbstractNumericTask,
        partitions: NumericPartitions,
        numeric_domain_sizes: Vec<usize>,
        combine_labels: bool,
    ) -> Result<Self> {
        let num_vars = task.get_num_variables() as usize;
        let derived_prop: HashSet<u32> = task
            .comparison_axioms()
            .iter()
            .map(|ax| ax.get_affected_var_id() as u32)
            .collect();

        let mut domain_mapping: DomainMapping = Vec::with_capacity(num_vars);
        let mut domain_sizes: Vec<i32> = Vec::with_capacity(num_vars);
        for var_id in 0..num_vars {
            if derived_prop.contains(&(var_id as u32)) {
                domain_mapping.push(vec![0, 1, 2]);
                domain_sizes.push(3);
            } else {
                let size_i32 = task
                    .get_variable_domain_size(var_id as i32)
                    .map_err(|e| anyhow!(e.to_string()))
                    .with_context(|| format!("failed to get domain size for variable {var_id}"))?;
                ensure!(
                    size_i32 > 0,
                    "non-positive domain size for variable {var_id}: {size_i32}"
                );
                let size = size_i32 as usize;
                let mapping: Vec<i32> = (0..size as i32).collect();
                domain_mapping.push(mapping);
                domain_sizes.push(size as i32);
            }
        }

        Self::new(
            task,
            domain_mapping,
            domain_sizes,
            partitions,
            numeric_domain_sizes,
            combine_labels,
        )
    }

    pub fn hash_multipliers(&self) -> &[i32] {
        &self.hash_multipliers
    }

    pub fn domain_sizes(&self) -> &[i32] {
        &self.domain_sizes
    }

    pub fn domain_mapping(&self) -> &DomainMapping {
        &self.domain_mapping
    }

    pub fn numeric_domain_sizes(&self) -> &[usize] {
        &self.numeric_domain_sizes
    }

    pub fn build_abstract_operators(
        &mut self,
        task: &dyn AbstractNumericTask,
    ) -> Result<Vec<AbstractOperator>> {
        let mut out: Vec<AbstractOperator> = Vec::new();
        let mut grouping: HashMap<OperatorSignature, usize> = HashMap::new();

        for (concrete_op_id, op) in task.get_operators().iter().enumerate() {
            self.build_for_concrete_operator(task, op, concrete_op_id, &mut out, &mut grouping)?;
        }

        Ok(out)
    }

    fn build_for_concrete_operator(
        &mut self,
        task: &dyn AbstractNumericTask,
        op: &Operator,
        concrete_op_id: usize,
        out: &mut Vec<AbstractOperator>,
        grouping: &mut HashMap<OperatorSignature, usize>,
    ) -> Result<()> {
        let (unconditional_effects, conditional_effects): (Vec<&Effect>, Vec<&Effect>) =
            op.effects().iter().partition(|e| e.conditions().is_empty());

        let (unconditional_ass, conditional_ass): (Vec<&AssignmentEffect>, Vec<&AssignmentEffect>) =
            op.assignment_effects()
                .iter()
                .partition(|e| !e.is_conditional());

        ensure!(
            conditional_effects.is_empty() && conditional_ass.is_empty(),
            "numeric-fd parity: conditional propositional or numeric effects are unsupported in abstract operator generation"
        );

        for eff in op.assignment_effects() {
            let rhs_var_id = eff.var_id() as usize;
            ensure!(
                rhs_var_id < task.numeric_variables().len(),
                "assignment effect rhs var id out of bounds: {} >= {}",
                rhs_var_id,
                task.numeric_variables().len()
            );
            ensure!(
                task.numeric_variables()[rhs_var_id].get_type() == &NumericType::Constant,
                "numeric-fd parity: assignment effects require constant RHS, got {:?} for numeric var {}",
                task.numeric_variables()[rhs_var_id].get_type(),
                rhs_var_id
            );
        }

        let ass_effects = op.assignment_effects().clone();
        build_branch_for_operator(
            task,
            op,
            &unconditional_effects,
            &ass_effects,
            op.preconditions(),
            concrete_op_id,
            self,
            out,
            grouping,
        )?;

        Ok(())
    }

    #[inline]
    fn variable_is_trivial(&self, var_id: usize) -> bool {
        self.domain_mapping
            .get(var_id)
            .unwrap_or_else(|| panic!("variable_is_trivial: var_id {var_id} out of bounds"))
            .is_empty()
    }

    #[inline]
    fn abstract_value(&self, var_id: usize, concrete_value: usize) -> i32 {
        let mapping = self
            .domain_mapping
            .get(var_id)
            .unwrap_or_else(|| panic!("abstract_value: var_id {var_id} out of bounds"));
        *mapping.get(concrete_value).unwrap_or_else(|| {
            panic!(
                "abstract_value: concrete value {concrete_value} out of bounds for variable {var_id}"
            )
        })
    }
}

fn format_abstract_fact(
    task: &dyn AbstractNumericTask,
    generator: &AbstractOperatorGenerator,
    fact: &Fact,
) -> String {
    let num_props = generator.domain_sizes.len();
    let var_id = fact.var() as usize;
    if var_id < num_props {
        let var_name = task.get_variable_name(var_id as i32).unwrap_or("<unknown>");
        let concrete_size = task
            .get_variable_domain_size(var_id as i32)
            .unwrap_or(0)
            .max(0) as usize;
        let mapping = generator.domain_mapping.get(var_id);
        let mut mapped_concretes: Vec<String> = Vec::new();
        for concrete_val in 0..concrete_size {
            let Some(abs_val) = mapping.and_then(|m| m.get(concrete_val)).copied() else {
                continue;
            };
            if abs_val == fact.value() {
                mapped_concretes.push(
                    task.get_fact_name(&Fact::new(fact.var(), concrete_val as i32))
                        .to_string(),
                );
            }
        }
        if mapped_concretes.is_empty() {
            format!("var{var_id}({var_name})=abs{}", fact.value())
        } else {
            format!(
                "var{var_id}({var_name})=abs{} => [{}]",
                fact.value(),
                mapped_concretes.join(" | ")
            )
        }
    } else {
        let numeric_var_id = var_id - num_props;
        let var_name = task
            .numeric_variables()
            .get(numeric_var_id)
            .map(|v| v.name())
            .unwrap_or("<unknown>");
        let interval = usize::try_from(fact.value())
            .ok()
            .and_then(|pid| generator.partitions.partition_interval(numeric_var_id, pid));
        match interval {
            Some(iv) => format!(
                "num{numeric_var_id}({var_name})=p{}:{}",
                fact.value(),
                utils::fmt_interval(iv)
            ),
            None => format!("num{numeric_var_id}({var_name})=p{}", fact.value()),
        }
    }
}

fn normalize_preconditions(mut preconditions: Vec<Fact>) -> Option<Vec<Fact>> {
    preconditions.sort();
    let mut out: Vec<Fact> = Vec::with_capacity(preconditions.len());
    for pre in preconditions {
        if let Some(last) = out.last() {
            if last.var() == pre.var() {
                if last.value() != pre.value() {
                    return None;
                }
                continue;
            }
        }
        out.push(pre);
    }
    Some(out)
}

fn build_branch_for_operator(
    task: &dyn AbstractNumericTask,
    op: &Operator,
    effects: &[&Effect],
    ass_effects: &[AssignmentEffect],
    merged_preconditions: &[Fact],
    concrete_op_id: usize,
    generator: &mut AbstractOperatorGenerator,
    out: &mut Vec<AbstractOperator>,
    grouping: &mut HashMap<OperatorSignature, usize>,
) -> Result<()> {
    let abstract_cost = abstract_operator_cost(task, op);
    let num_variables = task.get_num_variables() as usize;
    let mut has_precondition_on_var: Vec<i32> = vec![-1; num_variables];
    let mut has_effect_on_var: Vec<i32> = vec![-1; num_variables];

    let mut prev_pairs: Vec<Fact> = Vec::new();
    let mut pre_pairs: Vec<Fact> = Vec::new();
    let mut eff_pairs: Vec<Fact> = Vec::new();
    let mut effects_without_pre: Vec<Fact> = Vec::new();

    for pre in merged_preconditions {
        let var_id = pre.var() as usize;
        if generator.variable_is_trivial(var_id) {
            has_precondition_on_var[var_id] = 0;
            continue;
        }
        let abs_val = generator.abstract_value(var_id, pre.value() as usize);
        has_precondition_on_var[var_id] = abs_val;
    }

    for eff in effects {
        let var_id = eff.var_id() as usize;
        if generator.variable_is_trivial(var_id) {
            continue;
        }

        debug_assert!(!generator.derived_prop_vars.contains(&eff.var_id()));

        let abs_val = generator.abstract_value(var_id, eff.value() as usize);
        let pre_val = has_precondition_on_var[var_id];
        if pre_val < 0 {
            effects_without_pre.push(Fact::new(var_id as u32, abs_val));
        } else if pre_val != abs_val {
            has_effect_on_var[var_id] = abs_val;
            eff_pairs.push(Fact::new(var_id as u32, abs_val));
        }
    }

    for pre in merged_preconditions {
        let var_id = pre.var() as usize;
        if generator.variable_is_trivial(var_id) {
            continue;
        }
        let abs_val = generator.abstract_value(var_id, pre.value() as usize);
        if has_effect_on_var[var_id] >= 0 {
            pre_pairs.push(Fact::new(var_id as u32, abs_val));
        } else if !generator.derived_prop_vars.contains(&(var_id as u32)) {
            prev_pairs.push(Fact::new(var_id as u32, abs_val));
        }
    }

    for pre in merged_preconditions {
        let var_id = pre.var() as usize;
        if generator.variable_is_trivial(var_id) {
            continue;
        }
        if generator.derived_prop_vars.contains(&(var_id as u32)) {
            let abs_val = generator.abstract_value(var_id, pre.value() as usize);
            pre_pairs.push(Fact::new(var_id as u32, abs_val));
            let unknown_abs = generator.abstract_value(var_id, COMPARISON_UNKNOWN_VAL);
            eff_pairs.push(Fact::new(var_id as u32, unknown_abs));
        }
    }

    multiply_out_propositional(
        0,
        abstract_cost,
        &mut prev_pairs,
        &mut pre_pairs,
        &mut eff_pairs,
        &effects_without_pre,
        ass_effects,
        merged_preconditions,
        concrete_op_id,
        task,
        generator,
        out,
        grouping,
    )?;

    Ok(())
}

fn abstract_operator_cost(task: &dyn AbstractNumericTask, op: &Operator) -> f64 {
    metric_operator_cost_from_initial_values(task, op)
}

fn multiply_out_propositional(
    pos: usize,
    cost: f64,
    prev_pairs: &mut Vec<Fact>,
    pre_pairs: &mut Vec<Fact>,
    eff_pairs: &mut Vec<Fact>,
    effects_without_pre: &[Fact],
    ass_effects: &[AssignmentEffect],
    op_preconditions: &[Fact],
    concrete_op_id: usize,
    task: &dyn AbstractNumericTask,
    generator: &mut AbstractOperatorGenerator,
    out: &mut Vec<AbstractOperator>,
    grouping: &mut HashMap<OperatorSignature, usize>,
) -> Result<()> {
    fn has_in_list_conflict(facts: &[Fact]) -> bool {
        facts
            .windows(2)
            .any(|w| w[0].var() == w[1].var() && w[0].value() != w[1].value())
    }

    fn has_cross_list_conflict(a: &[Fact], b: &[Fact]) -> bool {
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

    if pos == effects_without_pre.len() {
        if eff_pairs.is_empty() && ass_effects.is_empty() {
            return Ok(());
        }

        let transitions = compute_hash_effects_with_preconditions(
            task,
            generator,
            op_preconditions,
            ass_effects,
        )?;
        for trans in transitions {
            let mut extended_pre_pairs = pre_pairs.clone();
            let mut extended_eff_pairs = eff_pairs.clone();
            let mut extended_prev_pairs = prev_pairs.clone();

            // Avoid duplicating variables that already appear in pre_pairs.
            let mut vars_in_pre: HashSet<u32> = pre_pairs.iter().map(|f| f.var()).collect();

            let source_by_var: HashMap<u32, Fact> = trans
                .source_partition_facts
                .iter()
                .cloned()
                .map(|f| (f.var(), f))
                .collect();
            let target_by_var: HashMap<u32, Fact> = trans
                .target_partition_facts
                .iter()
                .cloned()
                .map(|f| (f.var(), f))
                .collect();

            let mut transition_vars: Vec<u32> = source_by_var
                .keys()
                .chain(target_by_var.keys())
                .copied()
                .collect();
            transition_vars.sort_unstable();
            transition_vars.dedup();

            for var_id in transition_vars {
                let source_fact = source_by_var.get(&var_id);
                let target_fact = target_by_var.get(&var_id);

                if !vars_in_pre.insert(var_id) {
                    continue;
                }

                if let Some(src) = source_fact {
                    extended_pre_pairs.push(src.clone());
                }
                if let Some(tgt) = target_fact {
                    extended_eff_pairs.push(tgt.clone());
                }
            }
            extended_prev_pairs.extend(trans.prevail_facts.iter().cloned());

            // Sorting is required by numeric-fd's invariants (unique vars per list).
            extended_pre_pairs.sort();
            extended_eff_pairs.sort();
            extended_prev_pairs.sort();

            // Remove exact duplicates and drop inconsistent operators early.
            extended_pre_pairs.dedup();
            extended_eff_pairs.dedup();
            extended_prev_pairs.dedup();

            if has_in_list_conflict(&extended_pre_pairs)
                || has_in_list_conflict(&extended_eff_pairs)
                || has_in_list_conflict(&extended_prev_pairs)
                || has_cross_list_conflict(&extended_pre_pairs, &extended_prev_pairs)
                || has_cross_list_conflict(&extended_prev_pairs, &extended_eff_pairs)
            {
                continue;
            }

            let signature = OperatorSignature {
                prev_pairs: extended_prev_pairs
                    .iter()
                    .map(|f| (f.var(), f.value()))
                    .collect(),
                pre_pairs: extended_pre_pairs
                    .iter()
                    .map(|f| (f.var(), f.value()))
                    .collect(),
                eff_pairs: extended_eff_pairs
                    .iter()
                    .map(|f| (f.var(), f.value()))
                    .collect(),
                cost_bits: cost.to_bits(),
            };

            if generator.combine_labels {
                if let Some(&idx) = grouping.get(&signature) {
                    out[idx].concrete_op_ids.push(concrete_op_id);
                    continue;
                }
            }

            let op = AbstractOperator::new(
                &extended_prev_pairs,
                &extended_pre_pairs,
                &extended_eff_pairs,
                cost,
                &generator.hash_multipliers,
                vec![concrete_op_id],
                trans.changed_numeric_vars,
            );

            let idx = out.len();
            out.push(op);
            if generator.combine_labels {
                grouping.insert(signature, idx);
            }
        }

        return Ok(());
    }

    let var_id = effects_without_pre[pos].var() as usize;
    let eff = effects_without_pre[pos].value();
    let domain_size = generator.domain_sizes[var_id] as i32;
    for i in 0..domain_size {
        if i != eff {
            pre_pairs.push(Fact::new(var_id as u32, i));
            eff_pairs.push(Fact::new(var_id as u32, eff));
        } else {
            prev_pairs.push(Fact::new(var_id as u32, i));
        }

        multiply_out_propositional(
            pos + 1,
            cost,
            prev_pairs,
            pre_pairs,
            eff_pairs,
            effects_without_pre,
            ass_effects,
            op_preconditions,
            concrete_op_id,
            task,
            generator,
            out,
            grouping,
        )?;

        if i != eff {
            pre_pairs.pop();
            eff_pairs.pop();
        } else {
            prev_pairs.pop();
        }
    }

    Ok(())
}

fn compute_hash_effects_with_preconditions(
    task: &dyn AbstractNumericTask,
    generator: &mut AbstractOperatorGenerator,
    op_preconditions: &[Fact],
    ass_effects: &[planners_sas::numeric::numeric_task::AssignmentEffect],
) -> Result<Vec<TransitionInfo>> {
    if generator.numeric_domain_sizes.is_empty() {
        return Ok(vec![TransitionInfo {
            source_partition_facts: Vec::new(),
            target_partition_facts: Vec::new(),
            prevail_facts: Vec::new(),
            changed_numeric_vars: Vec::new(),
        }]);
    }

    let num_props = generator.domain_sizes.len() as u32;

    // Enumerate transitions per refined numeric variable.
    // IMPORTANT: numeric-fd also enumerates identity transitions for refined-but-unaffected
    // numeric vars (frame-like constraints). Without this, abstract operators can become
    // overly optimistic (e.g., allowing `pour` regardless of x/y partitions).
    let num_numeric_vars = generator.numeric_domain_sizes.len();
    let mut effects_by_var: Vec<Vec<&planners_sas::numeric::numeric_task::AssignmentEffect>> =
        vec![Vec::new(); num_numeric_vars];
    for eff in ass_effects {
        let v = eff.affected_var_id() as usize;
        debug_assert!(
            v < effects_by_var.len(),
            "assignment effect affected_var_id out of bounds: {v} >= {}",
            effects_by_var.len()
        );
        if v >= effects_by_var.len() {
            continue;
        }
        effects_by_var[v].push(eff);
    }

    let mut per_var: Vec<(usize, Vec<(usize, usize)>)> = Vec::new();
    let mut affected_numeric_vars: HashSet<usize> = HashSet::new();
    for v in 0..num_numeric_vars {
        if task.numeric_variables()[v].get_type() == &NumericType::Derived {
            continue;
        }
        let num_parts = generator.numeric_domain_sizes[v];
        if num_parts <= 1 {
            continue;
        }

        let effs = &effects_by_var[v];
        if let Some(eff) = effs.first() {
            affected_numeric_vars.insert(v);
            let rhs = eff.var_id() as usize;
            let rhs_parts = generator
                .partitions
                .partitions(rhs)
                .map(|s| s.len())
                .ok_or_else(|| anyhow!("missing partitions for rhs numeric var {rhs}"))?;

            let mut pairs: HashSet<(usize, usize)> = HashSet::new();
            for src in 0..num_parts {
                for rhs_part in 0..rhs_parts {
                    let rhs_iv = generator
                        .partitions
                        .partition_interval(rhs, rhs_part)
                        .with_context(|| {
                            format!("missing partition interval for rhs var {rhs} part {rhs_part}")
                        })?;
                    let targets =
                        generator
                            .partitions
                            .reachable_partitions(v, src, eff.operation(), rhs_iv);
                    for tgt in targets {
                        pairs.insert((src, tgt));
                    }
                }
            }
            let mut transitions: Vec<(usize, usize)> = pairs.into_iter().collect();
            transitions.sort_unstable();
            per_var.push((v, transitions));
        } else {
            // Unaffected refined numeric var: enumerate identity transitions p -> p.
            let transitions: Vec<(usize, usize)> = (0..num_parts).map(|p| (p, p)).collect();
            per_var.push((v, transitions));
        }
    }

    if per_var.is_empty() {
        return Ok(vec![TransitionInfo {
            source_partition_facts: Vec::new(),
            target_partition_facts: Vec::new(),
            prevail_facts: Vec::new(),
            changed_numeric_vars: Vec::new(),
        }]);
    }

    // Cartesian product across numeric variables.
    let mut combos: Vec<Vec<(usize, usize, usize)>> = vec![Vec::new()];
    for (var_id, transitions) in per_var {
        let mut next: Vec<Vec<(usize, usize, usize)>> = Vec::new();
        for prefix in &combos {
            for (src, tgt) in &transitions {
                let mut v = prefix.clone();
                v.push((var_id, *src, *tgt));
                next.push(v);
            }
        }
        combos = next;
        if combos.is_empty() {
            break;
        }
    }

    let mut out: Vec<TransitionInfo> = Vec::new();
    for combo in combos {
        let mut source_partition_facts: Vec<Fact> = Vec::new();
        let mut target_partition_facts: Vec<Fact> = Vec::new();
        let mut prevail_facts: Vec<Fact> = Vec::new();

        let mut changed_numeric_vars: Vec<usize> = Vec::new();

        for (var_id, src, tgt) in &combo {
            let abs_var_id = num_props + (*var_id as u32);
            source_partition_facts.push(Fact::new(abs_var_id, *src as i32));
            target_partition_facts.push(Fact::new(abs_var_id, *tgt as i32));
            if affected_numeric_vars.contains(var_id) {
                changed_numeric_vars.push(*var_id);
            }
        }

        // Note: we do NOT encode explicit effects on comparison/derived vars here.
        // They are treated as implicitly (re-)evaluated from numeric intervals during
        // regression/predecessor enumeration.

        // Optimistic filtering for comparison preconditions based on *source* intervals.
        let source_numeric_intervals =
            build_numeric_intervals_for_combo(task, generator, &combo, false)?;
        if let Some(index) = &generator.comparison_index {
            if op_preconditions
                .iter()
                .any(|pre| index.precondition_is_contradicted(pre, &source_numeric_intervals))
            {
                continue;
            }
        }

        if !changed_numeric_vars.is_empty() {
            let mut old_partitions: Vec<usize> = Vec::with_capacity(changed_numeric_vars.len());
            let mut new_partitions: Vec<usize> = Vec::with_capacity(changed_numeric_vars.len());
            for (var_id, src, tgt) in &combo {
                if affected_numeric_vars.contains(var_id) {
                    old_partitions.push(*src);
                    new_partitions.push(*tgt);
                }
            }

            source_partition_facts.extend(compute_direct_comparison_cascades(
                task,
                generator,
                &changed_numeric_vars,
                &old_partitions,
                &new_partitions,
                &old_partitions,
            )?);
            target_partition_facts.extend(compute_direct_comparison_cascades(
                task,
                generator,
                &changed_numeric_vars,
                &old_partitions,
                &new_partitions,
                &new_partitions,
            )?);

            source_partition_facts.extend(compute_assignment_axiom_cascades(
                task, generator, &combo, true,
            )?);
            target_partition_facts.extend(compute_assignment_axiom_cascades(
                task, generator, &combo, false,
            )?);
        }

        source_partition_facts.sort();
        source_partition_facts.dedup();
        target_partition_facts.sort();
        target_partition_facts.dedup();
        changed_numeric_vars.sort_unstable();
        changed_numeric_vars.dedup();

        out.push(TransitionInfo {
            source_partition_facts,
            target_partition_facts,
            prevail_facts,
            changed_numeric_vars,
        });
    }

    Ok(out)
}

fn build_numeric_intervals_for_combo(
    task: &dyn AbstractNumericTask,
    generator: &AbstractOperatorGenerator,
    combo: &[(usize, usize, usize)],
    use_target_partitions: bool,
) -> Result<Vec<Interval>> {
    let initial_numeric_values = task.get_initial_numeric_state_values();
    let mut numeric_intervals: Vec<Interval> =
        vec![Interval::new(0.0, 0.0, false, false); task.numeric_variables().len()];

    for (var_id, numeric_var) in task.numeric_variables().iter().enumerate() {
        if numeric_var.get_type() == &NumericType::Constant {
            numeric_intervals[var_id] = Interval::singleton(initial_numeric_values[var_id]);
        } else if numeric_var.get_type() != &NumericType::Derived
            && generator
                .numeric_domain_sizes
                .get(var_id)
                .copied()
                .unwrap_or(0)
                == 1
        {
            numeric_intervals[var_id] = Interval::unbounded();
        }
    }

    for (var_id, src, tgt) in combo {
        let partition_id = if use_target_partitions { *tgt } else { *src };
        let iv = generator
            .partitions
            .partition_interval(*var_id, partition_id)
            .with_context(|| {
                format!("missing partition interval for var {var_id} part {partition_id}")
            })?;
        numeric_intervals[*var_id] = iv;
    }

    propagate_assignment_axiom_intervals_known_only(task, &mut numeric_intervals);

    for interval in &mut numeric_intervals {
        if interval.is_empty() {
            *interval = Interval::unbounded();
        }
    }

    Ok(numeric_intervals)
}

fn propagate_assignment_axiom_intervals_known_only(
    task: &dyn AbstractNumericTask,
    numeric_intervals: &mut [Interval],
) {
    let max_iterations = task.assignment_axioms().len().saturating_add(1).max(1);
    for _ in 0..max_iterations {
        let mut changed = false;
        for axiom in task.assignment_axioms() {
            let Ok(affected_var_id) = usize::try_from(axiom.get_affected_var_id()) else {
                continue;
            };
            let Ok(left_var_id) = usize::try_from(axiom.get_left_var_id()) else {
                continue;
            };
            let Ok(right_var_id) = usize::try_from(axiom.get_right_var_id()) else {
                continue;
            };
            if affected_var_id >= numeric_intervals.len()
                || left_var_id >= numeric_intervals.len()
                || right_var_id >= numeric_intervals.len()
            {
                continue;
            }

            let lhs = numeric_intervals[left_var_id];
            let rhs = numeric_intervals[right_var_id];
            if lhs.is_empty() || rhs.is_empty() {
                continue;
            }

            let next = arith_op_from_axiom(axiom.get_operator()).apply_interval(lhs, rhs);
            if numeric_intervals[affected_var_id] != next {
                numeric_intervals[affected_var_id] = next;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
}

fn tri_value_for_comparison(
    tree: &ComparisonTree,
    inputs: &[Interval],
    affected_var_id: usize,
    domain_mapping: &DomainMapping,
) -> i32 {
    match tree.evaluate_interval(inputs) {
        Some(true) => domain_mapping[affected_var_id][COMPARISON_TRUE_VAL],
        Some(false) => domain_mapping[affected_var_id][COMPARISON_FALSE_VAL],
        None => domain_mapping[affected_var_id][COMPARISON_UNKNOWN_VAL],
    }
}

fn compute_direct_comparison_cascades(
    task: &dyn AbstractNumericTask,
    generator: &AbstractOperatorGenerator,
    changed_numeric_vars: &[usize],
    old_partitions: &[usize],
    new_partitions: &[usize],
    eval_partitions: &[usize],
) -> Result<Vec<Fact>> {
    let old_by_var: HashMap<usize, usize> = changed_numeric_vars
        .iter()
        .copied()
        .zip(old_partitions.iter().copied())
        .collect();
    let new_by_var: HashMap<usize, usize> = changed_numeric_vars
        .iter()
        .copied()
        .zip(new_partitions.iter().copied())
        .collect();
    let eval_by_var: HashMap<usize, usize> = changed_numeric_vars
        .iter()
        .copied()
        .zip(eval_partitions.iter().copied())
        .collect();

    let mut affected_facts: Vec<Fact> = Vec::new();
    for comparison_axiom in task.comparison_axioms() {
        let left_var_id = comparison_axiom.get_left_var_id() as usize;
        let right_var_id = comparison_axiom.get_right_var_id() as usize;
        if !old_by_var.contains_key(&left_var_id) && !old_by_var.contains_key(&right_var_id) {
            continue;
        }

        let left_old = old_by_var.get(&left_var_id).copied();
        let left_new = new_by_var.get(&left_var_id).copied();
        let right_old = old_by_var.get(&right_var_id).copied();
        let right_new = new_by_var.get(&right_var_id).copied();

        let partition_changed = left_old != left_new || right_old != right_new;
        if !partition_changed {
            continue;
        }

        let Some(eval_left_partition) = eval_by_var
            .get(&left_var_id)
            .copied()
            .or_else(|| constant_partition(task, left_var_id))
        else {
            continue;
        };
        let Some(eval_right_partition) = eval_by_var
            .get(&right_var_id)
            .copied()
            .or_else(|| constant_partition(task, right_var_id))
        else {
            continue;
        };

        let Some(left_interval) = generator
            .partitions
            .partition_interval(left_var_id, eval_left_partition)
        else {
            continue;
        };
        let Some(right_interval) = generator
            .partitions
            .partition_interval(right_var_id, eval_right_partition)
        else {
            continue;
        };

        let affected_var_id = comparison_axiom.get_affected_var_id() as usize;
        match apply_comparison_interval(
            comparison_axiom.get_operator(),
            left_interval,
            right_interval,
        ) {
            Some(true) => affected_facts.push(Fact::new(
                affected_var_id as u32,
                generator.domain_mapping[affected_var_id][COMPARISON_TRUE_VAL],
            )),
            Some(false) => affected_facts.push(Fact::new(
                affected_var_id as u32,
                generator.domain_mapping[affected_var_id][COMPARISON_FALSE_VAL],
            )),
            None => {}
        }
    }

    Ok(affected_facts)
}

fn constant_partition(task: &dyn AbstractNumericTask, var_id: usize) -> Option<usize> {
    let numeric_var = task.numeric_variables().get(var_id)?;
    (numeric_var.get_type() == &NumericType::Constant).then_some(0)
}

fn compute_assignment_axiom_cascades(
    task: &dyn AbstractNumericTask,
    generator: &AbstractOperatorGenerator,
    combo: &[(usize, usize, usize)],
    evaluate_old_partitions: bool,
) -> Result<Vec<Fact>> {
    let old_by_var: HashMap<usize, usize> = combo
        .iter()
        .map(|(var_id, src, _)| (*var_id, *src))
        .collect();
    let new_by_var: HashMap<usize, usize> = combo
        .iter()
        .map(|(var_id, _, tgt)| (*var_id, *tgt))
        .collect();

    let mut derived_changed_vars: Vec<usize> = Vec::new();
    let mut derived_old_partitions: Vec<usize> = Vec::new();
    let mut derived_new_partitions: Vec<usize> = Vec::new();

    for axiom in task.assignment_axioms() {
        let derived_var_id = axiom.get_affected_var_id() as usize;
        let left_var_id = axiom.get_left_var_id() as usize;
        let right_var_id = axiom.get_right_var_id() as usize;

        let Some(left_old_partition) = old_by_var
            .get(&left_var_id)
            .copied()
            .or_else(|| constant_partition(task, left_var_id))
        else {
            continue;
        };
        let Some(right_old_partition) = old_by_var
            .get(&right_var_id)
            .copied()
            .or_else(|| constant_partition(task, right_var_id))
        else {
            continue;
        };
        let Some(left_new_partition) = new_by_var
            .get(&left_var_id)
            .copied()
            .or_else(|| constant_partition(task, left_var_id))
        else {
            continue;
        };
        let Some(right_new_partition) = new_by_var
            .get(&right_var_id)
            .copied()
            .or_else(|| constant_partition(task, right_var_id))
        else {
            continue;
        };

        if left_old_partition == left_new_partition && right_old_partition == right_new_partition {
            continue;
        }

        let Some(left_old_interval) = generator
            .partitions
            .partition_interval(left_var_id, left_old_partition)
        else {
            continue;
        };
        let Some(right_old_interval) = generator
            .partitions
            .partition_interval(right_var_id, right_old_partition)
        else {
            continue;
        };
        let Some(left_new_interval) = generator
            .partitions
            .partition_interval(left_var_id, left_new_partition)
        else {
            continue;
        };
        let Some(right_new_interval) = generator
            .partitions
            .partition_interval(right_var_id, right_new_partition)
        else {
            continue;
        };

        let old_range = arith_op_from_axiom(axiom.get_operator())
            .apply_interval(left_old_interval, right_old_interval);
        let new_range = arith_op_from_axiom(axiom.get_operator())
            .apply_interval(left_new_interval, right_new_interval);

        let Some(derived_partitions) = generator.partitions.partitions(derived_var_id) else {
            continue;
        };
        let old_derived_partition = derived_partitions
            .iter()
            .position(|partition| intervals_overlap(*partition, old_range));
        let new_derived_partition = derived_partitions
            .iter()
            .position(|partition| intervals_overlap(*partition, new_range));

        let (Some(old_derived_partition), Some(new_derived_partition)) =
            (old_derived_partition, new_derived_partition)
        else {
            continue;
        };
        if old_derived_partition == new_derived_partition {
            continue;
        }

        derived_changed_vars.push(derived_var_id);
        derived_old_partitions.push(old_derived_partition);
        derived_new_partitions.push(new_derived_partition);
    }

    compute_direct_comparison_cascades(
        task,
        generator,
        &derived_changed_vars,
        &derived_old_partitions,
        &derived_new_partitions,
        if evaluate_old_partitions {
            &derived_old_partitions
        } else {
            &derived_new_partitions
        },
    )
}

fn apply_comparison_interval(
    op: &ComparisonOperator,
    lhs: Interval,
    rhs: Interval,
) -> Option<bool> {
    if lhs.is_empty() || rhs.is_empty() {
        return Some(false);
    }

    let (lmin, lmin_c) = (lhs.lower, lhs.lower_closed);
    let (lmax, lmax_c) = (lhs.upper, lhs.upper_closed);
    let (rmin, rmin_c) = (rhs.lower, rhs.lower_closed);
    let (rmax, rmax_c) = (rhs.upper, rhs.upper_closed);

    let max_lt_min = |amax: f64, amax_c: bool, bmin: f64, bmin_c: bool| -> bool {
        (amax < bmin) || (amax == bmin && (!amax_c || !bmin_c))
    };
    let min_ge_max = |amin: f64, amin_c: bool, bmax: f64, bmax_c: bool| -> bool {
        (amin > bmax) || (amin == bmax && (amin_c && bmax_c))
    };
    let min_gt_max = |amin: f64, amin_c: bool, bmax: f64, bmax_c: bool| -> bool {
        (amin > bmax) || (amin == bmax && (!amin_c || !bmax_c))
    };
    let intervals_are_disjoint =
        || max_lt_min(lmax, lmax_c, rmin, rmin_c) || max_lt_min(rmax, rmax_c, lmin, lmin_c);

    match op {
        ComparisonOperator::LessThan => {
            if max_lt_min(lmax, lmax_c, rmin, rmin_c) {
                Some(true)
            } else if min_ge_max(lmin, lmin_c, rmax, rmax_c) {
                Some(false)
            } else {
                None
            }
        }
        ComparisonOperator::LessThanOrEqual => {
            if lmax <= rmin {
                Some(true)
            } else if min_gt_max(lmin, lmin_c, rmax, rmax_c) {
                Some(false)
            } else {
                None
            }
        }
        ComparisonOperator::GreaterThan => {
            apply_comparison_interval(opposite_comparison(op), rhs, lhs)
        }
        ComparisonOperator::GreaterThanOrEqual => {
            apply_comparison_interval(opposite_comparison(op), rhs, lhs)
        }
        ComparisonOperator::Equal => {
            if lhs.lower == lhs.upper
                && lhs.lower_closed
                && lhs.upper_closed
                && rhs.lower == rhs.upper
                && rhs.lower_closed
                && rhs.upper_closed
                && lmin == rmin
            {
                Some(true)
            } else if intervals_are_disjoint() {
                Some(false)
            } else {
                None
            }
        }
        ComparisonOperator::UnEqual => {
            if lhs.lower == lhs.upper
                && lhs.lower_closed
                && lhs.upper_closed
                && rhs.lower == rhs.upper
                && rhs.lower_closed
                && rhs.upper_closed
                && lmin == rmin
            {
                Some(false)
            } else if intervals_are_disjoint() {
                Some(true)
            } else {
                None
            }
        }
    }
}

fn opposite_comparison(op: &ComparisonOperator) -> &ComparisonOperator {
    match op {
        ComparisonOperator::GreaterThan => &ComparisonOperator::LessThan,
        ComparisonOperator::GreaterThanOrEqual => &ComparisonOperator::LessThanOrEqual,
        _ => op,
    }
}

fn intervals_overlap(a: Interval, b: Interval) -> bool {
    if a.is_empty() || b.is_empty() {
        return false;
    }

    if (a.upper < b.lower) || (a.upper == b.lower && (!a.upper_closed || !b.lower_closed)) {
        return false;
    }

    if (b.upper < a.lower) || (b.upper == a.lower && (!b.upper_closed || !a.lower_closed)) {
        return false;
    }

    true
}

fn comparison_dependency_partition_changed(
    task: &dyn AbstractNumericTask,
    tree: &ComparisonTree,
    combo: &[(usize, usize, usize)],
) -> bool {
    tree.regular_numeric_var_dependencies(task)
        .into_iter()
        .filter_map(|var_id| usize::try_from(var_id).ok())
        .any(|var_id| {
            combo
                .iter()
                .any(|(changed_var_id, source_partition, target_partition)| {
                    *changed_var_id == var_id && source_partition != target_partition
                })
        })
}

fn compute_hash_multipliers(
    domain_sizes: &[i32],
    numeric_domain_sizes: &[usize],
) -> Result<Vec<i32>> {
    let mut multipliers: Vec<i32> =
        Vec::with_capacity(domain_sizes.len() + numeric_domain_sizes.len());
    let mut num_states: i64 = 1;

    for &size in domain_sizes {
        ensure!(size > 0, "domain size must be > 0, got {size}");
        ensure!(
            num_states <= i64::from(i32::MAX),
            "hash multiplier overflow (too many abstract states)"
        );
        multipliers.push(num_states as i32);
        num_states = num_states
            .checked_mul(size as i64)
            .ok_or_else(|| anyhow!("hash multiplier overflow (too many abstract states)"))?;
    }
    for &parts in numeric_domain_sizes {
        ensure!(parts > 0, "numeric domain size must be > 0");
        ensure!(
            num_states <= i64::from(i32::MAX),
            "hash multiplier overflow (too many abstract states)"
        );
        multipliers.push(num_states as i32);
        num_states = num_states
            .checked_mul(parts as i64)
            .ok_or_else(|| anyhow!("hash multiplier overflow (too many abstract states)"))?;
    }
    Ok(multipliers)
}
