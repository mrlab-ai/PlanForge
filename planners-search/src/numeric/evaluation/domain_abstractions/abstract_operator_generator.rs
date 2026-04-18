#[cfg(test)]
mod tests;

use std::collections::{HashMap, HashSet};
use std::hash::Hash;

use anyhow::{Context, Result, anyhow, ensure};

use planners_sas::numeric::axioms::CalOperator;

use planners_sas::numeric::numeric_task::{
    AbstractNumericTask, AssignmentEffect, Effect, ExplicitFact, NumericType, Operator,
    metric_operator_cost_from_initial_values,
};

use super::comparison_expression::{ArithOp, ComparisonTree, Interval};
use super::domain_abstraction::{ComparisonAxiomIndex, NumericPartitions};
use super::numeric_context::fill_derived_numeric_intervals_from_comparison_trees;
use super::utils;

const COMPARISON_TRUE_VAL: usize = 0;
const COMPARISON_FALSE_VAL: usize = 1;
const COMPARISON_UNKNOWN_VAL: usize = 2;

pub type DomainMapping = Vec<Vec<usize>>;

#[derive(Debug, Clone, PartialEq)]
pub struct AbstractOperator {
    pub concrete_op_ids: Vec<usize>,
    pub cost: f64,
    pub hash_effect: i32,
    pub regression_preconditions: Vec<ExplicitFact>,
    pub preconditions: Vec<ExplicitFact>,
    pub changed_numeric_vars: Vec<usize>,
}

impl AbstractOperator {
    pub fn new(
        prev_pairs: &[ExplicitFact],
        pre_pairs: &[ExplicitFact],
        eff_pairs: &[ExplicitFact],
        cost: f64,
        hash_multipliers: &[usize],
        concrete_op_ids: Vec<usize>,
        changed_numeric_vars: Vec<usize>,
    ) -> Self {
        let mut preconditions: Vec<ExplicitFact> = pre_pairs.to_vec();
        preconditions.extend_from_slice(prev_pairs);
        preconditions.sort();
        debug_assert!(preconditions.windows(2).all(|w| w[0].var != w[1].var));

        let mut regression_preconditions: Vec<ExplicitFact> = prev_pairs.to_vec();
        regression_preconditions.extend_from_slice(eff_pairs);
        regression_preconditions.sort();
        debug_assert!(
            regression_preconditions
                .windows(2)
                .all(|w| w[0].var != w[1].var)
        );

        debug_assert_eq!(
            pre_pairs.len(),
            eff_pairs.len(),
            "abstract operator pre/eff pair mismatch: pre_pairs={pre_pairs:?} eff_pairs={eff_pairs:?}"
        );

        let mut hash_effect: i32 = 0;
        for (pre, eff) in pre_pairs.iter().zip(eff_pairs.iter()) {
            debug_assert_eq!(
                pre.var, eff.var,
                "abstract operator transition var mismatch: pre={pre:?} eff={eff:?}"
            );

            let var = pre.var;
            let multiplier = hash_multipliers[var];
            let new_val = pre.value as i32;
            let old_val = eff.value as i32;
            hash_effect += (new_val - old_val) * multiplier as i32;
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
    pub source_partition_facts: Vec<ExplicitFact>,
    pub target_partition_facts: Vec<ExplicitFact>,
    pub prevail_facts: Vec<ExplicitFact>,
    pub changed_numeric_vars: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct OperatorSignature {
    prev_pairs: Vec<(usize, usize)>,
    pre_pairs: Vec<(usize, usize)>,
    eff_pairs: Vec<(usize, usize)>,
    cost_bits: u64,
}

#[allow(unused)]
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
    domain_sizes: Vec<usize>,
    numeric_domain_sizes: Vec<usize>,
    hash_multipliers: Vec<usize>,
    partitions: NumericPartitions,
    comparison_index: Option<ComparisonAxiomIndex>,
    comparison_trees: Vec<ComparisonTree>,
    comparisons_by_numeric_dep: Vec<Vec<usize>>,
    derived_prop_vars: HashSet<usize>,
    combine_labels: bool,
}

impl AbstractOperatorGenerator {
    pub fn new(
        task: &dyn AbstractNumericTask,
        domain_mapping: DomainMapping,
        domain_sizes: Vec<usize>,
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
                ensure!(
                    dep < comparisons_by_numeric_dep.len(),
                    "comparison tree depends on numeric var {dep}, but only {} numeric vars exist",
                    comparisons_by_numeric_dep.len()
                );
                comparisons_by_numeric_dep[dep].push(tree_idx);
            }
        }

        let derived_prop_vars: HashSet<usize> = task
            .comparison_axioms()
            .iter()
            .map(|ax| ax.get_affected_var_id())
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
        let num_vars = task.get_num_variables();
        let derived_prop: HashSet<usize> = task
            .comparison_axioms()
            .iter()
            .map(|ax| ax.get_affected_var_id())
            .collect();

        let mut domain_mapping: DomainMapping = Vec::with_capacity(num_vars);
        let mut domain_sizes: Vec<usize> = Vec::with_capacity(num_vars);
        for var_id in 0..num_vars {
            if derived_prop.contains(&var_id) {
                domain_mapping.push(vec![
                    COMPARISON_TRUE_VAL,
                    COMPARISON_FALSE_VAL,
                    COMPARISON_UNKNOWN_VAL,
                ]);
                domain_sizes.push(3);
            } else {
                let size = task
                    .get_variable_domain_size(var_id)
                    .map_err(|e| anyhow!(e.to_string()))
                    .with_context(|| format!("failed to get domain size for variable {var_id}"))?;
                let mapping: Vec<usize> = (0..size).collect();
                domain_mapping.push(mapping);
                domain_sizes.push(size);
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

    pub fn hash_multipliers(&self) -> &[usize] {
        &self.hash_multipliers
    }

    pub fn domain_sizes(&self) -> &[usize] {
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

        let (_unconditional_ass, conditional_ass): (
            Vec<&AssignmentEffect>,
            Vec<&AssignmentEffect>,
        ) = op
            .assignment_effects()
            .iter()
            .partition(|e| !e.is_conditional());

        ensure!(
            conditional_effects.is_empty() && conditional_ass.is_empty(),
            "numeric-fd parity: conditional propositional or numeric effects are unsupported in abstract operator generation"
        );

        for eff in op.assignment_effects() {
            let rhs_var_id = eff.var_id();
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
        self.domain_sizes
            .get(var_id)
            .copied()
            .unwrap_or_else(|| panic!("variable_is_trivial: var_id {var_id} out of bounds"))
            <= 1
    }

    #[inline]
    fn abstract_value(&self, var_id: usize, concrete_value: usize) -> usize {
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

#[allow(unused)]
fn format_abstract_fact(
    task: &dyn AbstractNumericTask,
    generator: &AbstractOperatorGenerator,
    fact: &ExplicitFact,
) -> String {
    let num_props = generator.domain_sizes.len();
    let var_id = fact.var;
    if var_id < num_props {
        let var_name = task.get_variable_name(var_id).unwrap_or("<unknown>");
        let concrete_size = task.get_variable_domain_size(var_id).unwrap_or(0);
        let mapping = generator.domain_mapping.get(var_id);
        let mut mapped_concretes: Vec<String> = Vec::new();
        for concrete_val in 0..concrete_size {
            let Some(abs_val) = mapping.and_then(|m| m.get(concrete_val)).copied() else {
                continue;
            };
            if abs_val == fact.value {
                mapped_concretes.push(
                    task.get_fact_name(&ExplicitFact::new(fact.var, concrete_val))
                        .to_string(),
                );
            }
        }
        if mapped_concretes.is_empty() {
            format!("var{var_id}({var_name})=abs{}", fact.value)
        } else {
            format!(
                "var{var_id}({var_name})=abs{} => [{}]",
                fact.value,
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
        let interval = generator
            .partitions
            .partition_interval(numeric_var_id, fact.value);
        match interval {
            Some(iv) => format!(
                "num{numeric_var_id}({var_name})=p{}:{}",
                fact.value,
                utils::fmt_interval(iv)
            ),
            None => format!("num{numeric_var_id}({var_name})=p{}", fact.value),
        }
    }
}

#[allow(unused)]
fn normalize_preconditions(mut preconditions: Vec<ExplicitFact>) -> Option<Vec<ExplicitFact>> {
    preconditions.sort();
    let mut out: Vec<ExplicitFact> = Vec::with_capacity(preconditions.len());
    for pre in preconditions {
        if let Some(last) = out.last()
            && last.var == pre.var
        {
            if last.value != pre.value {
                return None;
            }
            continue;
        }
        out.push(pre);
    }
    Some(out)
}

#[allow(clippy::too_many_arguments)]
fn build_branch_for_operator(
    task: &dyn AbstractNumericTask,
    op: &Operator,
    effects: &[&Effect],
    ass_effects: &[AssignmentEffect],
    merged_preconditions: &[ExplicitFact],
    concrete_op_id: usize,
    generator: &mut AbstractOperatorGenerator,
    out: &mut Vec<AbstractOperator>,
    grouping: &mut HashMap<OperatorSignature, usize>,
) -> Result<()> {
    let abstract_cost = abstract_operator_cost(task, op);
    let num_variables = task.get_num_variables();
    let mut precondition_on_var: Vec<Option<usize>> = vec![None; num_variables];
    let mut effect_on_var: Vec<Option<usize>> = vec![None; num_variables];

    let mut prev_pairs: Vec<ExplicitFact> = Vec::new();
    let mut pre_pairs: Vec<ExplicitFact> = Vec::new();
    let mut eff_pairs: Vec<ExplicitFact> = Vec::new();
    let mut effects_without_pre: Vec<ExplicitFact> = Vec::new();

    for pre in merged_preconditions {
        let var_id = pre.var;
        if generator.variable_is_trivial(var_id) {
            precondition_on_var[var_id] = Some(0);
            continue;
        }
        let abs_val = generator.abstract_value(var_id, pre.value);
        precondition_on_var[var_id] = Some(abs_val);
    }

    for eff in effects {
        let var_id = eff.var_id();
        if generator.variable_is_trivial(var_id) {
            continue;
        }

        debug_assert!(!generator.derived_prop_vars.contains(&eff.var_id()));

        let abs_val = generator.abstract_value(var_id, eff.value());
        let pre = precondition_on_var[var_id];
        if let Some(pre_val) = pre {
            if pre_val != abs_val {
                effect_on_var[var_id] = Some(abs_val);
                eff_pairs.push(ExplicitFact::new(var_id, abs_val));
            }
        } else {
            effects_without_pre.push(ExplicitFact::new(var_id, abs_val));
        }
    }

    for pre in merged_preconditions {
        let var_id = pre.var;
        if generator.variable_is_trivial(var_id) {
            continue;
        }
        let abs_val = generator.abstract_value(var_id, pre.value);
        if effect_on_var[var_id].is_some() {
            pre_pairs.push(ExplicitFact::new(var_id, abs_val));
        } else if !generator.derived_prop_vars.contains(&(var_id)) {
            prev_pairs.push(ExplicitFact::new(var_id, abs_val));
        }
    }

    for pre in merged_preconditions {
        let var_id = pre.var;
        if generator.variable_is_trivial(var_id) {
            continue;
        }
        if generator.derived_prop_vars.contains(&(var_id)) {
            let abs_val = generator.abstract_value(var_id, pre.value);
            pre_pairs.push(ExplicitFact::new(var_id, abs_val));
            let unknown_abs = generator.abstract_value(var_id, COMPARISON_UNKNOWN_VAL);
            eff_pairs.push(ExplicitFact::new(var_id, unknown_abs));
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

#[allow(clippy::too_many_arguments)]
fn multiply_out_propositional(
    pos: usize,
    cost: f64,
    prev_pairs: &mut Vec<ExplicitFact>,
    pre_pairs: &mut Vec<ExplicitFact>,
    eff_pairs: &mut Vec<ExplicitFact>,
    effects_without_pre: &[ExplicitFact],
    ass_effects: &[AssignmentEffect],
    op_preconditions: &[ExplicitFact],
    concrete_op_id: usize,
    task: &dyn AbstractNumericTask,
    generator: &mut AbstractOperatorGenerator,
    out: &mut Vec<AbstractOperator>,
    grouping: &mut HashMap<OperatorSignature, usize>,
) -> Result<()> {
    fn has_in_list_conflict(facts: &[ExplicitFact]) -> bool {
        facts
            .windows(2)
            .any(|w| w[0].var == w[1].var && w[0].value != w[1].value)
    }

    fn has_cross_list_conflict(a: &[ExplicitFact], b: &[ExplicitFact]) -> bool {
        let mut i = 0;
        let mut j = 0;
        while i < a.len() && j < b.len() {
            let va = a[i].var;
            let vb = b[j].var;
            if va < vb {
                i += 1;
            } else if vb < va {
                j += 1;
            } else {
                if a[i].value != b[j].value {
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
            let mut vars_in_pre: HashSet<usize> = pre_pairs.iter().map(|f| f.var).collect();

            let source_by_var: HashMap<usize, ExplicitFact> = trans
                .source_partition_facts
                .iter()
                .cloned()
                .map(|f| (f.var, f))
                .collect();
            let target_by_var: HashMap<usize, ExplicitFact> = trans
                .target_partition_facts
                .iter()
                .cloned()
                .map(|f| (f.var, f))
                .collect();

            let mut transition_vars: Vec<usize> = source_by_var
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
                    .map(|f| (f.var, f.value))
                    .collect(),
                pre_pairs: extended_pre_pairs
                    .iter()
                    .map(|f| (f.var, f.value))
                    .collect(),
                eff_pairs: extended_eff_pairs
                    .iter()
                    .map(|f| (f.var, f.value))
                    .collect(),
                cost_bits: cost.to_bits(),
            };

            if generator.combine_labels
                && let Some(&idx) = grouping.get(&signature)
            {
                out[idx].concrete_op_ids.push(concrete_op_id);
                continue;
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

    let var_id = effects_without_pre[pos].var;
    let eff = effects_without_pre[pos].value;
    let domain_size = generator.domain_sizes[var_id];
    for i in 0..domain_size {
        if i != eff {
            pre_pairs.push(ExplicitFact::new(var_id, i));
            eff_pairs.push(ExplicitFact::new(var_id, eff));
        } else {
            prev_pairs.push(ExplicitFact::new(var_id, i));
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

#[allow(clippy::needless_range_loop)]
fn compute_hash_effects_with_preconditions(
    task: &dyn AbstractNumericTask,
    generator: &mut AbstractOperatorGenerator,
    op_preconditions: &[ExplicitFact],
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

    let num_props = generator.domain_sizes.len();

    // Enumerate transitions per refined numeric variable.
    // IMPORTANT: numeric-fd also enumerates identity transitions for refined-but-unaffected
    // numeric vars (frame-like constraints). Without this, abstract operators can become
    // overly optimistic (e.g., allowing `pour` regardless of x/y partitions).
    let num_numeric_vars = generator.numeric_domain_sizes.len();
    let mut effects_by_var: Vec<Vec<&planners_sas::numeric::numeric_task::AssignmentEffect>> =
        vec![Vec::new(); num_numeric_vars];
    for eff in ass_effects {
        let v = eff.affected_var_id();
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
            let rhs = eff.var_id();
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
            // Unaffected refined numeric var: enumerate identity transitions `p` -> `p`.
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
        let mut source_partition_facts: Vec<ExplicitFact> = Vec::new();
        let mut target_partition_facts: Vec<ExplicitFact> = Vec::new();
        let prevail_facts: Vec<ExplicitFact> = Vec::new();

        let mut changed_numeric_vars: Vec<usize> = Vec::new();

        for (var_id, src, tgt) in &combo {
            let abs_var_id = num_props + (*var_id);
            source_partition_facts.push(ExplicitFact::new(abs_var_id, *src));
            target_partition_facts.push(ExplicitFact::new(abs_var_id, *tgt));
            if affected_numeric_vars.contains(var_id) {
                changed_numeric_vars.push(*var_id);
            }
        }

        // Note: we do NOT encode explicit effects on comparison/derived vars here.
        // They are treated as implicitly (re-)evaluated from numeric intervals during
        // regression/predecessor enumeration.

        // Optimistic filtering for comparison preconditions based on *source* intervals.
        let source_numeric_intervals =
            prepare_comparison_tree_inputs_for_combo(task, generator, &combo, false)?;
        if let Some(index) = &generator.comparison_index
            && op_preconditions
                .iter()
                .any(|pre| index.precondition_is_contradicted(pre, &source_numeric_intervals))
        {
            continue;
        }

        if !changed_numeric_vars.is_empty() {
            let (source_comparison_facts, target_comparison_facts) =
                compute_comparison_tree_cascades(task, generator, &combo)?;
            source_partition_facts.extend(source_comparison_facts);
            target_partition_facts.extend(target_comparison_facts);
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

fn prepare_comparison_tree_inputs_for_combo(
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

    fill_derived_numeric_intervals_from_comparison_trees(
        &generator.comparison_trees,
        &mut numeric_intervals,
    );

    for interval in &mut numeric_intervals {
        if interval.is_empty() {
            *interval = Interval::unbounded();
        }
    }

    Ok(numeric_intervals)
}

fn tri_value_for_comparison(
    tree: &ComparisonTree,
    inputs: &[Interval],
    affected_var_id: usize,
    domain_mapping: &DomainMapping,
) -> usize {
    match tree.evaluate_interval(inputs) {
        Some(true) => domain_mapping[affected_var_id][COMPARISON_TRUE_VAL],
        Some(false) => domain_mapping[affected_var_id][COMPARISON_FALSE_VAL],
        None => domain_mapping[affected_var_id][COMPARISON_UNKNOWN_VAL],
    }
}

fn compute_comparison_tree_cascades(
    task: &dyn AbstractNumericTask,
    generator: &AbstractOperatorGenerator,
    combo: &[(usize, usize, usize)],
) -> Result<(Vec<ExplicitFact>, Vec<ExplicitFact>)> {
    let source_inputs = prepare_comparison_tree_inputs_for_combo(task, generator, combo, false)?;
    let target_inputs = prepare_comparison_tree_inputs_for_combo(task, generator, combo, true)?;

    let mut affected_tree_ids: Vec<usize> = Vec::new();
    let mut seen_tree_ids: HashSet<usize> = HashSet::new();
    for (var_id, source_partition, target_partition) in combo {
        if source_partition == target_partition {
            continue;
        }
        let tree_ids = generator
            .comparisons_by_numeric_dep
            .get(*var_id)
            .with_context(|| {
                format!("missing comparison dependency bucket for numeric var {var_id}")
            })?;
        for &tree_id in tree_ids {
            if seen_tree_ids.insert(tree_id) {
                affected_tree_ids.push(tree_id);
            }
        }
    }

    let mut source_facts: Vec<ExplicitFact> = Vec::new();
    let mut target_facts: Vec<ExplicitFact> = Vec::new();
    for tree_id in affected_tree_ids {
        let tree = generator
            .comparison_trees
            .get(tree_id)
            .with_context(|| format!("missing comparison tree {tree_id}"))?;
        ensure!(
            tree.affected_var_id < generator.domain_mapping.len(),
            "comparison tree {tree_id} affected var {1} out of range for domain mapping of len {0}",
            generator.domain_mapping.len(),
            tree.affected_var_id
        );

        let source_value = tri_value_for_comparison(
            tree,
            &source_inputs,
            tree.affected_var_id,
            &generator.domain_mapping,
        );
        let target_value = tri_value_for_comparison(
            tree,
            &target_inputs,
            tree.affected_var_id,
            &generator.domain_mapping,
        );

        let unknown_value = generator.domain_mapping[tree.affected_var_id][COMPARISON_UNKNOWN_VAL];
        if source_value == target_value
            || source_value == unknown_value
            || target_value == unknown_value
        {
            continue;
        }

        source_facts.push(ExplicitFact::new(tree.affected_var_id, source_value));
        target_facts.push(ExplicitFact::new(tree.affected_var_id, target_value));
    }

    Ok((source_facts, target_facts))
}

#[allow(unused)]
fn comparison_dependency_partition_changed(
    task: &dyn AbstractNumericTask,
    tree: &ComparisonTree,
    combo: &[(usize, usize, usize)],
) -> bool {
    tree.regular_numeric_var_dependencies(task)
        .into_iter()
        .any(|var_id| {
            combo
                .iter()
                .any(|(changed_var_id, source_partition, target_partition)| {
                    *changed_var_id == var_id && source_partition != target_partition
                })
        })
}

fn compute_hash_multipliers(
    domain_sizes: &[usize],
    numeric_domain_sizes: &[usize],
) -> Result<Vec<usize>> {
    let mut multipliers: Vec<usize> =
        Vec::with_capacity(domain_sizes.len() + numeric_domain_sizes.len());
    let mut num_states: usize = 1;

    for &size in domain_sizes {
        multipliers.push(num_states);
        num_states = num_states
            .checked_mul(size)
            .ok_or_else(|| anyhow!("hash multiplier overflow (too many abstract states)"))?;
    }
    for &parts in numeric_domain_sizes {
        multipliers.push(num_states);
        num_states = num_states
            .checked_mul(parts)
            .ok_or_else(|| anyhow!("hash multiplier overflow (too many abstract states)"))?;
    }
    Ok(multipliers)
}
