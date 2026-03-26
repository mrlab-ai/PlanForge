#[cfg(test)]
mod tests;

use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};

use anyhow::{Context, Result, anyhow, ensure};

use planners_sas::numeric::numeric_task::{
    AbstractNumericTask, AssignmentEffect, Effect, Fact, NumericType, Operator,
};

use super::comparison_expression::{ComparisonTree, Interval};
use super::domain_abstraction::{ComparisonAxiomIndex, NumericPartitions};

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
        // We can end up emitting the same fact twice when building abstract operators.
        // This is benign, but duplicate facts for the same variable with different
        // values would be a bug (kept guarded by the assertion below).
        preconditions.dedup();
        debug_assert!(preconditions.windows(2).all(|w| w[0].var() != w[1].var()));

        let mut regression_preconditions: Vec<Fact> = prev_pairs.to_vec();
        regression_preconditions.extend_from_slice(eff_pairs);
        regression_preconditions.sort();
        regression_preconditions.dedup();
        debug_assert!(
            regression_preconditions
                .windows(2)
                .all(|w| w[0].var() != w[1].var())
        );

        let mut hash_effect: i32 = 0;
        let mut pre_idx = 0usize;
        let mut eff_idx = 0usize;
        while pre_idx < pre_pairs.len() && eff_idx < eff_pairs.len() {
            let pre = &pre_pairs[pre_idx];
            let eff = &eff_pairs[eff_idx];
            if pre.var() < eff.var() {
                pre_idx += 1;
                continue;
            }
            if eff.var() < pre.var() {
                eff_idx += 1;
                continue;
            }

            let var = pre.var() as usize;
            let multiplier = hash_multipliers[var];
            let new_val = pre.value();
            let old_val = eff.value();
            hash_effect += (new_val - old_val) * multiplier;
            pre_idx += 1;
            eff_idx += 1;
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

        if conditional_effects.is_empty() && conditional_ass.is_empty() {
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
            return Ok(());
        }

        let mut extra_preconditions: Vec<Fact> = Vec::new();
        let mut chosen_conditional_effects: Vec<&Effect> = Vec::new();
        let mut chosen_conditional_ass: Vec<&AssignmentEffect> = Vec::new();

        enumerate_conditional_propositional_effects(
            0,
            task,
            op,
            &unconditional_effects,
            &conditional_effects,
            &unconditional_ass,
            &conditional_ass,
            &mut extra_preconditions,
            &mut chosen_conditional_effects,
            &mut chosen_conditional_ass,
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
        let mapping = &self.domain_mapping[var_id];
        mapping[concrete_value]
    }
}

fn enumerate_conditional_propositional_effects<'a>(
    idx: usize,
    task: &dyn AbstractNumericTask,
    op: &'a Operator,
    unconditional_effects: &[&'a Effect],
    conditional_effects: &[&'a Effect],
    unconditional_ass: &[&'a AssignmentEffect],
    conditional_ass: &[&'a AssignmentEffect],
    extra_preconditions: &mut Vec<Fact>,
    chosen_conditional_effects: &mut Vec<&'a Effect>,
    chosen_conditional_ass: &mut Vec<&'a AssignmentEffect>,
    concrete_op_id: usize,
    generator: &mut AbstractOperatorGenerator,
    out: &mut Vec<AbstractOperator>,
    grouping: &mut HashMap<OperatorSignature, usize>,
) -> Result<()> {
    if idx == conditional_effects.len() {
        enumerate_conditional_assignment_effects(
            0,
            task,
            op,
            unconditional_effects,
            unconditional_ass,
            conditional_ass,
            extra_preconditions,
            chosen_conditional_effects,
            chosen_conditional_ass,
            concrete_op_id,
            generator,
            out,
            grouping,
        )?;
        return Ok(());
    }

    // Branch: conditional effect not applied.
    enumerate_conditional_propositional_effects(
        idx + 1,
        task,
        op,
        unconditional_effects,
        conditional_effects,
        unconditional_ass,
        conditional_ass,
        extra_preconditions,
        chosen_conditional_effects,
        chosen_conditional_ass,
        concrete_op_id,
        generator,
        out,
        grouping,
    )?;

    // Branch: conditional effect applied.
    let eff = conditional_effects[idx];
    let n = eff.conditions().len();
    extra_preconditions.extend_from_slice(eff.conditions());
    chosen_conditional_effects.push(eff);
    enumerate_conditional_propositional_effects(
        idx + 1,
        task,
        op,
        unconditional_effects,
        conditional_effects,
        unconditional_ass,
        conditional_ass,
        extra_preconditions,
        chosen_conditional_effects,
        chosen_conditional_ass,
        concrete_op_id,
        generator,
        out,
        grouping,
    )?;
    chosen_conditional_effects.pop();
    extra_preconditions.truncate(extra_preconditions.len() - n);

    Ok(())
}

fn enumerate_conditional_assignment_effects<'a>(
    idx: usize,
    task: &dyn AbstractNumericTask,
    op: &'a Operator,
    unconditional_effects: &[&'a Effect],
    unconditional_ass: &[&'a AssignmentEffect],
    conditional_ass: &[&'a AssignmentEffect],
    extra_preconditions: &mut Vec<Fact>,
    chosen_conditional_effects: &mut Vec<&'a Effect>,
    chosen_conditional_ass: &mut Vec<&'a AssignmentEffect>,
    concrete_op_id: usize,
    generator: &mut AbstractOperatorGenerator,
    out: &mut Vec<AbstractOperator>,
    grouping: &mut HashMap<OperatorSignature, usize>,
) -> Result<()> {
    if idx == conditional_ass.len() {
        let mut branch_effects: Vec<&Effect> =
            Vec::with_capacity(unconditional_effects.len() + chosen_conditional_effects.len());
        branch_effects.extend_from_slice(unconditional_effects);
        branch_effects.extend(chosen_conditional_effects.iter().copied());

        let mut ass_effects: Vec<AssignmentEffect> =
            Vec::with_capacity(unconditional_ass.len() + chosen_conditional_ass.len());
        ass_effects.extend(unconditional_ass.iter().copied().cloned());
        ass_effects.extend(chosen_conditional_ass.iter().copied().cloned());

        let mut merged_preconditions: Vec<Fact> = op.preconditions().to_vec();
        merged_preconditions.extend_from_slice(extra_preconditions);
        let Some(merged_preconditions) = normalize_preconditions(merged_preconditions) else {
            return Ok(());
        };

        build_branch_for_operator(
            task,
            op,
            &branch_effects,
            &ass_effects,
            &merged_preconditions,
            concrete_op_id,
            generator,
            out,
            grouping,
        )?;
        return Ok(());
    }

    // Branch: conditional assignment effect not applied.
    enumerate_conditional_assignment_effects(
        idx + 1,
        task,
        op,
        unconditional_effects,
        unconditional_ass,
        conditional_ass,
        extra_preconditions,
        chosen_conditional_effects,
        chosen_conditional_ass,
        concrete_op_id,
        generator,
        out,
        grouping,
    )?;

    // Branch: conditional assignment effect applied.
    let eff = conditional_ass[idx];
    let n = eff.conditions().len();
    extra_preconditions.extend_from_slice(eff.conditions());
    chosen_conditional_ass.push(eff);
    enumerate_conditional_assignment_effects(
        idx + 1,
        task,
        op,
        unconditional_effects,
        unconditional_ass,
        conditional_ass,
        extra_preconditions,
        chosen_conditional_effects,
        chosen_conditional_ass,
        concrete_op_id,
        generator,
        out,
        grouping,
    )?;
    chosen_conditional_ass.pop();
    extra_preconditions.truncate(extra_preconditions.len() - n);

    Ok(())
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
        op.cost() as f64,
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

#[derive(Clone, Debug, PartialEq, Eq)]
struct OperatorSignature {
    prev_pairs: Vec<(u32, i32)>,
    pre_pairs: Vec<(u32, i32)>,
    eff_pairs: Vec<(u32, i32)>,
    cost_bits: u64,
}

impl Hash for OperatorSignature {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.prev_pairs.hash(state);
        self.pre_pairs.hash(state);
        self.eff_pairs.hash(state);
        self.cost_bits.hash(state);
    }
}

fn multiply_out_propositional(
    pos: usize,
    cost: f64,
    prev_pairs: &mut Vec<Fact>,
    pre_pairs: &mut Vec<Fact>,
    eff_pairs: &mut Vec<Fact>,
    effects_without_pre: &[Fact],
    ass_effects: &[planners_sas::numeric::numeric_task::AssignmentEffect],
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
        if v < effects_by_var.len() {
            effects_by_var[v].push(eff);
        }
    }

    let mut per_var: Vec<(usize, Vec<(usize, usize)>)> = Vec::new();
    for v in 0..num_numeric_vars {
        let num_parts = generator.numeric_domain_sizes[v];
        if num_parts <= 1 {
            continue;
        }

        let effs = &effects_by_var[v];
        ensure!(
            effs.len() <= 1,
            "multiple assignment effects on the same numeric var are not supported (var={v}, count={})",
            effs.len()
        );

        if let Some(eff) = effs.first() {
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
                    let targets = generator
                        .partitions
                        .reachable_partitions(v, src, eff.operation(), rhs_iv);
                    for tgt in targets {
                        pairs.insert((src, tgt));
                    }
                }
            }
            per_var.push((v, pairs.into_iter().collect()));
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
            if src == tgt {
                prevail_facts.push(Fact::new(abs_var_id, *src as i32));
            } else {
                source_partition_facts.push(Fact::new(abs_var_id, *src as i32));
                target_partition_facts.push(Fact::new(abs_var_id, *tgt as i32));
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
            let source_numeric_intervals =
                build_numeric_intervals_for_combo(task, generator, &combo, false)?;
            let target_numeric_intervals =
                build_numeric_intervals_for_combo(task, generator, &combo, true)?;
            let mut seen_comparisons: HashSet<usize> = HashSet::new();
            for &numeric_var_id in &changed_numeric_vars {
                let Some(tree_ids) = generator.comparisons_by_numeric_dep.get(numeric_var_id) else {
                    continue;
                };
                for &tree_id in tree_ids {
                    if !seen_comparisons.insert(tree_id) {
                        continue;
                    }

                    let tree = generator.comparison_trees.get(tree_id).with_context(|| {
                        format!("comparison tree id out of bounds while building operators: {tree_id}")
                    })?;
                    let affected_var_id = usize::try_from(tree.affected_var_id).with_context(|| {
                        format!(
                            "comparison tree affected_var_id does not fit usize: {}",
                            tree.affected_var_id
                        )
                    })?;
                    let source_abs = tri_value_for_comparison(
                        tree,
                        &source_numeric_intervals,
                        affected_var_id,
                        &generator.domain_mapping,
                    );
                    let target_abs = tri_value_for_comparison(
                        tree,
                        &target_numeric_intervals,
                        affected_var_id,
                        &generator.domain_mapping,
                    );
                    let unknown_abs = generator.domain_mapping[affected_var_id]
                        [COMPARISON_UNKNOWN_VAL];
                    if source_abs != unknown_abs
                        && target_abs != unknown_abs
                        && source_abs != target_abs
                    {
                        source_partition_facts
                            .push(Fact::new(affected_var_id as u32, source_abs));
                        target_partition_facts
                            .push(Fact::new(affected_var_id as u32, target_abs));
                    }
                }
            }
        }

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
    let mut numeric_intervals: Vec<Interval> = vec![Interval::unbounded(); task.numeric_variables().len()];

    for (i, v) in task.numeric_variables().iter().enumerate() {
        if v.get_type() == &NumericType::Constant {
            numeric_intervals[i] = Interval::singleton(initial_numeric_values[i]);
        }
    }

    for (var_id, src, tgt) in combo {
        let partition_id = if use_target_partitions { *tgt } else { *src };
        let iv = generator
            .partitions
            .partition_interval(*var_id, partition_id)
            .with_context(|| {
                format!(
                    "missing partition interval for var {var_id} part {partition_id}"
                )
            })?;
        numeric_intervals[*var_id] = iv;
    }

    Ok(numeric_intervals)
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
