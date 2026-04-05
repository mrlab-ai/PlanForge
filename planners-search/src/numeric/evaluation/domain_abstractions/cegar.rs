#[cfg(test)]
mod tests;

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt::{self, Write as _};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, ensure};
use rand::seq::SliceRandom;
use rand::{SeedableRng, rngs::SmallRng};
use serde::{Deserialize, Serialize};

use libc::exit;
use planners_sas::numeric::axioms::AxiomEvaluator;
use planners_sas::numeric::numeric_task::{AbstractNumericTask, ExplicitFact};
use planners_sas::numeric::utils::int_packer::IntDoublePacker;

use super::abstract_operator_generator::DomainMapping;
use super::comparison_expression::Interval;
use super::domain_abstraction::ComparisonAxiomIndex;
use super::domain_abstraction::NumericPartitions;
use super::domain_abstraction_factory::{DomainAbstractionFactory, WildcardPlanResult};

/// Mirrors numeric-fd's `NumericFlaw = tuple<int, ap_float, bool>`.
#[derive(Debug, Clone, PartialEq)]
pub struct NumericFlaw {
    pub numeric_var_id: usize,
    pub value: f64,
    pub include_in_lower: bool,
}

/// Mirrors numeric-fd's `PropFlaw = pair<Fact, vector<NumericFlaw>>`.
#[derive(Debug, Clone, PartialEq)]
pub struct PropFlaw {
    pub fact: ExplicitFact,
    pub dependent_numeric_flaws: Vec<NumericFlaw>,
}

/// Mirrors numeric-fd's `Flaw = variant<PropFlaw, NumericFlaw>`.
#[derive(Debug, Clone, PartialEq)]
pub enum Flaw {
    Propositional(PropFlaw),
    Numeric(NumericFlaw),
}

/// How `fix_flaws` chooses which flaws to refine.
///
/// This mirrors numeric-fd's `FlawTreatment` options, but our defaults aim to
/// stay deterministic.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FlawTreatment {
    RandomSingleAtom,
    OneSplitPerAtom,
    OneSplitPerVariable,
    MaxRefinedSingleAtom,
}

impl fmt::Display for FlawTreatment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RandomSingleAtom => write!(f, "random_single_atom"),
            Self::OneSplitPerAtom => write!(f, "one_split_per_atom"),
            Self::OneSplitPerVariable => write!(f, "one_split_per_variable"),
            Self::MaxRefinedSingleAtom => write!(f, "max_refined_single_atom"),
        }
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum InitSplitMethod {
    GoalValue,
    GoalValueOrRandomIfNonGoal,
    InitValue,
    RandomValue,
    RandomPartition,
    RandomBinaryPartitionSeparatingInitGoal,
    Identity,
}

impl fmt::Display for InitSplitMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::GoalValue => write!(f, "goal_value"),
            Self::GoalValueOrRandomIfNonGoal => write!(f, "goal_value_or_random_if_non_goal"),
            Self::InitValue => write!(f, "init_value"),
            Self::RandomValue => write!(f, "random_value"),
            Self::RandomPartition => write!(f, "random_partition"),
            Self::RandomBinaryPartitionSeparatingInitGoal => {
                write!(f, "random_binary_partition_separating_init_goal")
            }
            Self::Identity => write!(f, "identity"),
        }
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ExecEntirePlanMode {
    StopAtFirstFlaw,
    ExecuteEntirePlan,
}

impl fmt::Display for ExecEntirePlanMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StopAtFirstFlaw => write!(f, "stop_at_first_flaw"),
            Self::ExecuteEntirePlan => write!(f, "execute_entire_plan"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DependentNumericRefinement {
    None,
    One,
    All,
}

#[derive(Debug, Clone)]
pub struct CegarConfig {
    pub max_abstraction_size: usize,
    pub max_iterations: usize,
    pub max_time: Option<Duration>,
    pub use_wildcard_plans: bool, // TODO: Right now must be true, add config to factory
    pub combine_labels: bool,
    pub debug: bool,
    pub flaw_treatment: FlawTreatment,
    pub init_split_method: InitSplitMethod,
    pub exec_entire_plan: ExecEntirePlanMode,
    pub init_split_var_ids: Option<HashSet<usize>>,
}

impl Default for CegarConfig {
    fn default() -> Self {
        Self {
            max_abstraction_size: usize::MAX,
            max_iterations: 10_000,
            max_time: None,
            use_wildcard_plans: true,
            combine_labels: false,
            debug: false,
            flaw_treatment: FlawTreatment::RandomSingleAtom,
            init_split_method: InitSplitMethod::InitValue,
            exec_entire_plan: ExecEntirePlanMode::StopAtFirstFlaw,
            init_split_var_ids: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CegarState {
    pub domain_mapping: DomainMapping,
    pub domain_sizes: Vec<usize>,
    pub partitions: NumericPartitions,
    pub numeric_domain_sizes: Vec<usize>,
    pub iteration: usize,
}

#[derive(Debug, Clone)]
pub struct CegarStep {
    pub factory: DomainAbstractionFactory,
    pub wildcard_plan: Option<WildcardPlanResult>,
}

#[derive(Debug, Clone)]
pub struct CegarOutcome {
    pub final_state: CegarState,
    pub last_step: CegarStep,
}

#[derive(Debug, Clone)]
pub struct Cegar {
    config: CegarConfig,
}

impl Cegar {
    pub fn new(config: CegarConfig) -> Result<Self> {
        ensure!(
            config.max_abstraction_size > 0,
            "max_abstraction_size must be > 0"
        );
        ensure!(config.max_iterations > 0, "max_iterations must be > 0");
        Ok(Self { config })
    }

    pub fn build_abstraction(&self, task: &dyn AbstractNumericTask) -> Result<CegarOutcome> {
        run_cegar(task, self.config.clone())
    }

    pub fn get_flaws(
        &self,
        task: &dyn AbstractNumericTask,
        partitions: &NumericPartitions,
        wildcard_plan: &WildcardPlanResult,
        execute_entire_plan: bool,
    ) -> Result<Vec<Flaw>> {
        let comparison_index = ComparisonAxiomIndex::from_task(task)
            .map_err(|e| anyhow::anyhow!("failed to build ComparisonAxiomIndex: {e}"))?;

        let state_packer = make_prop_state_packer(task);
        let axiom_evaluator = AxiomEvaluator::new(task, &state_packer);

        let mut buffer = vec![0u64; state_packer.num_bins() as usize];
        set_initial_prop_values(task, &state_packer, &mut buffer);
        let mut numeric_state: Vec<f64> = task.get_initial_numeric_state_values().to_vec();

        axiom_evaluator
            .evaluate_arithmetic_axioms(&mut numeric_state)
            .map_err(|e| {
                anyhow::anyhow!("failed to evaluate arithmetic axioms for initial state: {e:?}")
            })?;
        axiom_evaluator
            .evaluate(&mut buffer, &mut numeric_state)
            .map_err(|e| anyhow::anyhow!("failed to evaluate axioms for initial state: {e:?}"))?;

        let mut step_flaws: Vec<Flaw> = Vec::new();
        let mut collected_flaws: Vec<Flaw> = Vec::new();
        let mut step_num: usize = 1;

        for equivalent_ops in wildcard_plan.wildcard_plan.iter() {
            ensure!(
                step_num < wildcard_plan.abstract_numeric_states.len(),
                "WildcardPlanResult abstract_numeric_states too short for step {step_num}"
            );
            let expected_abs_numeric_state = &wildcard_plan.abstract_numeric_states[step_num];

            step_flaws.clear();

            if !execute_entire_plan {
                let mut applied = false;
                for &op_id in equivalent_ops.iter() {
                    let Some(op) = task.get_operators().get(op_id) else {
                        continue;
                    };
                    let operator_flaws = get_precondition_flaws(
                        task,
                        partitions,
                        &comparison_index,
                        op,
                        &state_packer,
                        &buffer,
                        &numeric_state,
                    );
                    if operator_flaws.is_empty() {
                        let mut candidate_buffer = buffer.clone();
                        let numeric_state_before_op = numeric_state.clone();
                        let mut candidate_numeric_state = numeric_state.clone();
                        apply_operator_to_state(
                            op,
                            &state_packer,
                            &mut candidate_buffer,
                            &mut candidate_numeric_state,
                        );
                        axiom_evaluator
                            .evaluate_arithmetic_axioms(&mut candidate_numeric_state)
                            .map_err(|e| {
                                anyhow::anyhow!(
                                    "failed to evaluate arithmetic axioms after operator: {e:?}"
                                )
                            })?;
                        axiom_evaluator
                            .evaluate(&mut candidate_buffer, &mut candidate_numeric_state)
                            .map_err(|e| {
                                anyhow::anyhow!("failed to evaluate axioms after operator: {e:?}")
                            })?;

                        let deviation_flaws = get_numeric_deviation_flaws(
                            op,
                            &numeric_state_before_op,
                            &candidate_numeric_state,
                            expected_abs_numeric_state,
                            partitions,
                        );
                        if deviation_flaws.is_empty() {
                            buffer = candidate_buffer;
                            numeric_state = candidate_numeric_state;
                            applied = true;
                            step_flaws.clear();
                            break;
                        } else {
                            step_flaws.extend(deviation_flaws);
                        }
                    } else {
                        step_flaws.extend(operator_flaws);
                    }
                }

                if !applied {
                    return Ok(step_flaws.clone());
                }
                step_num += 1;
                continue;
            }

            // execute_entire_plan mode: keep executing even if flaws are found.
            let mut chosen_op_id: Option<usize> = None;
            let mut fallback_op_id: Option<usize> = None;
            for &op_id in equivalent_ops.iter() {
                if task.get_operators().get(op_id).is_none() {
                    continue;
                }
                if fallback_op_id.is_none() {
                    fallback_op_id = Some(op_id);
                }
                let op = &task.get_operators()[op_id];
                let operator_flaws = get_precondition_flaws(
                    task,
                    partitions,
                    &comparison_index,
                    op,
                    &state_packer,
                    &buffer,
                    &numeric_state,
                );
                if operator_flaws.is_empty() {
                    chosen_op_id = Some(op_id);
                    break;
                } else {
                    step_flaws.extend(operator_flaws);
                }
            }

            if !step_flaws.is_empty() {
                collected_flaws.extend(step_flaws.drain(..));
            }

            let chosen = chosen_op_id.or(fallback_op_id);
            if let Some(op_id) = chosen {
                let op = &task.get_operators()[op_id];
                let numeric_state_before_op = numeric_state.clone();
                apply_operator_to_state(op, &state_packer, &mut buffer, &mut numeric_state);
                axiom_evaluator
                    .evaluate_arithmetic_axioms(&mut numeric_state)
                    .map_err(|e| {
                        anyhow::anyhow!(
                            "failed to evaluate arithmetic axioms after operator: {e:?}"
                        )
                    })?;
                axiom_evaluator
                    .evaluate(&mut buffer, &mut numeric_state)
                    .map_err(|e| {
                        anyhow::anyhow!("failed to evaluate axioms after operator: {e:?}")
                    })?;

                let deviation_flaws = get_numeric_deviation_flaws(
                    op,
                    &numeric_state_before_op,
                    &numeric_state,
                    expected_abs_numeric_state,
                    partitions,
                );
                if !deviation_flaws.is_empty() {
                    collected_flaws.extend(deviation_flaws);
                }
            }

            step_num += 1;
        }

        let goal_flaws = get_goal_flaws(
            task,
            partitions,
            &comparison_index,
            &state_packer,
            &buffer,
            &numeric_state,
        );
        if execute_entire_plan {
            collected_flaws.extend(goal_flaws);
            Ok(collected_flaws)
        } else {
            Ok(goal_flaws)
        }
    }

    /// Port of numeric-fd's refinement step (`fix_flaws`).
    ///
    /// Returns `true` if any refinement was applied.
    pub fn fix_flaws(
        &self,
        task: &dyn AbstractNumericTask,
        flaws: &[Flaw],
        domain_mapping: &mut DomainMapping,
        domain_sizes: &mut Vec<usize>,
        partitions: &mut NumericPartitions,
        numeric_domain_sizes: &mut Vec<usize>,
    ) -> Result<bool> {
        let comparison_var_ids: HashSet<usize> = task
            .comparison_axioms()
            .iter()
            .map(|ax| ax.get_affected_var_id())
            .collect();
        let abstraction_size = compute_abstraction_size(domain_sizes, numeric_domain_sizes);

        match self.config.flaw_treatment {
            FlawTreatment::RandomSingleAtom => fix_single_random_flaw(
                task,
                flaws,
                &self.config,
                &comparison_var_ids,
                domain_mapping,
                domain_sizes,
                partitions,
                numeric_domain_sizes,
                abstraction_size,
            ),
            FlawTreatment::OneSplitPerAtom => fix_flaws_per_atom(
                task,
                flaws,
                &self.config,
                &comparison_var_ids,
                domain_mapping,
                domain_sizes,
                partitions,
                numeric_domain_sizes,
                abstraction_size,
            ),
            FlawTreatment::OneSplitPerVariable => fix_flaws_per_variable(
                task,
                flaws,
                &self.config,
                &comparison_var_ids,
                domain_mapping,
                domain_sizes,
                partitions,
                numeric_domain_sizes,
                abstraction_size,
            ),
            FlawTreatment::MaxRefinedSingleAtom => fix_single_flaw_max_refined(
                task,
                flaws,
                &self.config,
                &comparison_var_ids,
                domain_mapping,
                domain_sizes,
                partitions,
                numeric_domain_sizes,
                abstraction_size,
            ),
        }
    }
}

fn compute_abstraction_size(domain_sizes: &[usize], numeric_domain_sizes: &[usize]) -> usize {
    let mut size: usize = 1;
    for &d in domain_sizes.iter() {
        if d == 0 {
            return 0;
        }
        size = size.saturating_mul(d);
    }
    for &p in numeric_domain_sizes.iter() {
        if p == 0 {
            return 0;
        }
        size = size.saturating_mul(p);
    }
    size
}

fn current_time_seed() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};

    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos() as u64)
        .unwrap_or(0x9e37_79b9_7f4a_7c15)
}

fn shuffle_indices_with_rng<R: rand::Rng + ?Sized>(indices: &mut [usize], rng: &mut R) {
    indices.shuffle(rng);
}

fn shuffle_indices(indices: &mut [usize]) {
    let mut rng = SmallRng::seed_from_u64(current_time_seed());
    shuffle_indices_with_rng(indices, &mut rng);
}

fn abstraction_size_u128(domain_sizes: &[usize], numeric_domain_sizes: &[usize]) -> Option<u128> {
    super::utils::compute_abstraction_size_u128(domain_sizes, numeric_domain_sizes)
}

fn can_refine_propositional_variable(
    domain_sizes: &[usize],
    numeric_domain_sizes: &[usize],
    var_id: usize,
    new_domain_size: usize,
    max_abstraction_size: usize,
) -> bool {
    let Some(total_size) = abstraction_size_u128(domain_sizes, numeric_domain_sizes) else {
        return false;
    };
    let Some(&old_domain_size) = domain_sizes.get(var_id) else {
        return false;
    };
    if old_domain_size == 0 || new_domain_size == 0 {
        return false;
    }
    let reduced = total_size / (old_domain_size as u128);
    reduced
        .checked_mul(new_domain_size as u128)
        .map(|candidate| candidate <= max_abstraction_size as u128)
        .unwrap_or(false)
}

fn can_refine_numeric_variable(
    domain_sizes: &[usize],
    numeric_domain_sizes: &[usize],
    numeric_var_id: usize,
    max_abstraction_size: usize,
) -> bool {
    let Some(total_size) = abstraction_size_u128(domain_sizes, numeric_domain_sizes) else {
        return false;
    };
    let Some(&old_partition_count) = numeric_domain_sizes.get(numeric_var_id) else {
        return false;
    };
    if old_partition_count == 0 {
        return false;
    }
    let reduced = total_size / (old_partition_count as u128);
    reduced
        .checked_mul((old_partition_count as u128) + 1)
        .map(|candidate| candidate <= max_abstraction_size as u128)
        .unwrap_or(false)
}

fn flaw_atom_key(flaw: &Flaw) -> (u8, usize, usize, u64, bool) {
    match flaw {
        Flaw::Propositional(pf) => (0, pf.fact.var, pf.fact.value, 0, false),
        Flaw::Numeric(nf) => (
            1,
            nf.numeric_var_id,
            0,
            nf.value.to_bits(),
            nf.include_in_lower,
        ),
    }
}

fn flaw_variable_key(flaw: &Flaw) -> (u8, usize) {
    match flaw {
        Flaw::Propositional(pf) => (0, pf.fact.var),
        Flaw::Numeric(nf) => (1, nf.numeric_var_id),
    }
}

fn score_flaw(
    flaw: &Flaw,
    domain_sizes: &[usize],
    numeric_domain_sizes: &[usize],
    _abstraction_size: usize,
) -> usize {
    match flaw {
        Flaw::Numeric(nf) => numeric_domain_sizes
            .get(nf.numeric_var_id)
            .copied()
            .unwrap_or(0),
        Flaw::Propositional(pf) => {
            let var_id = pf.fact.var;
            let base = domain_sizes.get(var_id).copied().unwrap_or(0);
            let max_dep = pf
                .dependent_numeric_flaws
                .iter()
                .filter_map(|nf| numeric_domain_sizes.get(nf.numeric_var_id).copied())
                .max()
                .unwrap_or(0);
            base + max_dep
        }
    }
}

fn fix_single_random_flaw(
    task: &dyn AbstractNumericTask,
    flaws: &[Flaw],
    config: &CegarConfig,
    comparison_var_ids: &HashSet<usize>,
    domain_mapping: &mut DomainMapping,
    domain_sizes: &mut Vec<usize>,
    partitions: &mut NumericPartitions,
    numeric_domain_sizes: &mut Vec<usize>,
    abstraction_size: usize,
) -> Result<bool> {
    if flaws.is_empty() {
        return Ok(false);
    }

    let mut indices: Vec<usize> = (0..flaws.len()).collect();
    shuffle_indices(&mut indices);

    let mut changed = false;
    let mut applied: usize = 0;
    let mut refined_prop_vars: HashSet<usize> = HashSet::new();
    let mut refined_numeric_vars: HashSet<usize> = HashSet::new();

    for idx in indices {
        if try_refine_from_flaw(
            task,
            &flaws[idx],
            config,
            comparison_var_ids,
            domain_mapping,
            domain_sizes,
            partitions,
            numeric_domain_sizes,
            DependentNumericRefinement::One,
        )? {
            return Ok(true);
        }
    }

    Ok(false)
}

fn fix_flaws_per_atom(
    task: &dyn AbstractNumericTask,
    flaws: &[Flaw],
    config: &CegarConfig,
    comparison_var_ids: &HashSet<usize>,
    domain_mapping: &mut DomainMapping,
    domain_sizes: &mut Vec<usize>,
    partitions: &mut NumericPartitions,
    numeric_domain_sizes: &mut Vec<usize>,
    _abstraction_size: usize,
) -> Result<bool> {
    let mut ordered: Vec<&Flaw> = flaws.iter().collect();
    ordered.sort_by(|a, b| flaw_atom_key(a).cmp(&flaw_atom_key(b)));

    let mut changed = false;
    let mut last: Option<(u8, usize, usize, u64, bool)> = None;
    for flaw in ordered {
        let key = flaw_atom_key(flaw);
        if last.as_ref() == Some(&key) {
            continue;
        }
        last = Some(key);
        let local_changed = try_refine_from_flaw(
            task,
            flaw,
            config,
            comparison_var_ids,
            domain_mapping,
            domain_sizes,
            partitions,
            numeric_domain_sizes,
            DependentNumericRefinement::All,
        )?;
        changed = changed || local_changed;
    }
    Ok(changed)
}

fn fix_flaws_per_variable(
    task: &dyn AbstractNumericTask,
    flaws: &[Flaw],
    config: &CegarConfig,
    comparison_var_ids: &HashSet<usize>,
    domain_mapping: &mut DomainMapping,
    domain_sizes: &mut Vec<usize>,
    partitions: &mut NumericPartitions,
    numeric_domain_sizes: &mut Vec<usize>,
    _abstraction_size: usize,
) -> Result<bool> {
    let mut ordered: Vec<&Flaw> = flaws.iter().collect();
    ordered.sort_by(|a, b| flaw_variable_key(a).cmp(&flaw_variable_key(b)));

    let mut changed = false;
    let mut refined_prop_vars: HashSet<usize> = HashSet::new();
    let mut refined_numeric_vars: HashSet<usize> = HashSet::new();
    let mut last: Option<(u8, usize)> = None;

    for flaw in ordered {
        let key = flaw_variable_key(flaw);
        if last.as_ref() == Some(&key) {
            continue;
        }
        last = Some(key);
        let local_changed = try_refine_from_flaw(
            task,
            flaw,
            config,
            comparison_var_ids,
            domain_mapping,
            domain_sizes,
            partitions,
            numeric_domain_sizes,
            DependentNumericRefinement::One,
        )?;
        changed = changed || local_changed;
    }
    Ok(changed)
}

fn fix_single_flaw_max_refined(
    task: &dyn AbstractNumericTask,
    flaws: &[Flaw],
    config: &CegarConfig,
    comparison_var_ids: &HashSet<usize>,
    domain_mapping: &mut DomainMapping,
    domain_sizes: &mut Vec<usize>,
    partitions: &mut NumericPartitions,
    numeric_domain_sizes: &mut Vec<usize>,
    abstraction_size: usize,
) -> Result<bool> {
    if flaws.is_empty() {
        return Ok(false);
    }

    #[derive(Clone)]
    struct Candidate {
        idx: usize,
        score: usize,
        restricted_dep: Option<Vec<NumericFlaw>>,
    }

    let mut candidates: Vec<Candidate> = Vec::with_capacity(flaws.len());
    for (idx, flaw) in flaws.iter().enumerate() {
        let mut restricted_dep: Option<Vec<NumericFlaw>> = None;
        let score: usize = match flaw {
            Flaw::Numeric(nf) => numeric_domain_sizes
                .get(nf.numeric_var_id)
                .copied()
                .unwrap_or(0),
            Flaw::Propositional(pf) => {
                let var_id = pf.fact.var;
                let base: usize = domain_sizes.get(var_id).copied().unwrap_or(0);
                if comparison_var_ids.contains(&var_id) && !pf.dependent_numeric_flaws.is_empty() {
                    let mut best: BTreeMap<usize, Vec<NumericFlaw>> = BTreeMap::new();
                    for nf in pf.dependent_numeric_flaws.iter().cloned() {
                        let partitions = numeric_domain_sizes
                            .get(nf.numeric_var_id)
                            .copied()
                            .unwrap_or(0);
                        best.entry(partitions).or_default().push(nf);
                    }
                    if let Some((&max_partitions, vec)) = best.iter().next_back() {
                        restricted_dep = Some(vec.clone());
                        base + (max_partitions)
                    } else {
                        base
                    }
                } else {
                    base
                }
            }
        };
        candidates.push(Candidate {
            idx,
            score,
            restricted_dep,
        });
    }

    // Highest score first; tie-break by stable atom key for determinism.
    candidates.sort_by(|a, b| {
        b.score
            .cmp(&a.score)
            .then_with(|| flaw_atom_key(&flaws[a.idx]).cmp(&flaw_atom_key(&flaws[b.idx])))
    });

    for cand in candidates {
        let mut chosen = flaws[cand.idx].clone();
        if let (Flaw::Propositional(pf), Some(restricted)) = (&mut chosen, cand.restricted_dep) {
            pf.dependent_numeric_flaws = restricted;
        }

        if try_refine_from_flaw(
            task,
            &chosen,
            config,
            comparison_var_ids,
            domain_mapping,
            domain_sizes,
            partitions,
            numeric_domain_sizes,
            DependentNumericRefinement::One,
        )? {
            return Ok(true);
        }
    }

    let _ = abstraction_size;
    Ok(false)
}

fn try_refine_from_flaw(
    task: &dyn AbstractNumericTask,
    flaw: &Flaw,
    config: &CegarConfig,
    comparison_var_ids: &HashSet<usize>,
    domain_mapping: &mut DomainMapping,
    domain_sizes: &mut [usize],
    partitions: &mut NumericPartitions,
    numeric_domain_sizes: &mut [usize],
    dependent_numeric_refinement: DependentNumericRefinement,
) -> Result<bool> {
    match flaw {
        Flaw::Numeric(nf) => {
            let var_id = nf.numeric_var_id;
            if !can_refine_numeric_variable(
                domain_sizes,
                numeric_domain_sizes,
                var_id,
                config.max_abstraction_size,
            ) {
                return Ok(false);
            }
            if partitions.split_at(var_id, nf.value, nf.include_in_lower) {
                if let Some(parts) = partitions.partitions(var_id) {
                    if let Some(slot) = numeric_domain_sizes.get_mut(var_id) {
                        *slot = parts.len();
                    }
                }
                return Ok(true);
            }
            Ok(false)
        }
        Flaw::Propositional(pf) => {
            let var_id = pf.fact.var;
            let value = pf.fact.value;

            // Bounds and conversion checks: these should hold in normal operation;
            // surface violations during debug builds but keep release behavior.
            if var_id >= domain_mapping.len() || var_id >= domain_sizes.len() {
                debug_assert!(
                    false,
                    "try_refine_from_flaw: var_id out of bounds: {} (mapping.len={}, domain_sizes.len={})",
                    var_id,
                    domain_mapping.len(),
                    domain_sizes.len()
                );
                return Ok(false);
            }

            let concrete_size = match task.get_variable_domain_size(var_id) {
                Ok(s) => s,
                Err(e) => {
                    debug_assert!(
                        false,
                        "try_refine_from_flaw: get_variable_domain_size({}) failed: {}",
                        var_id,
                        e.to_string()
                    );
                    return Ok(false);
                }
            };

            if value >= concrete_size {
                debug_assert!(
                    false,
                    "try_refine_from_flaw: fact value {} out of range (concrete size {}) for var {}",
                    value, concrete_size, var_id
                );
                return Ok(false);
            }

            let mut changed = false;

            if comparison_var_ids.contains(&var_id) {
                if !can_refine_propositional_variable(
                    domain_sizes,
                    numeric_domain_sizes,
                    var_id,
                    2,
                    config.max_abstraction_size,
                ) {
                    return Ok(false);
                }
                // Comparison axiom vars: split into {false/unknown} vs {true} like numeric-fd.
                let old_size = domain_sizes[var_id];
                if domain_sizes[var_id] < 2 {
                    domain_sizes[var_id] = 2;
                    changed = true;
                }
                // Ensure mapping values are within the new abstract size.
                if domain_mapping[var_id].len() >= 1 {
                    if domain_mapping[var_id][0] != 1 {
                        domain_mapping[var_id][0] = 1;
                        changed = true;
                    }
                }
                if domain_mapping[var_id].len() >= 2 {
                    if domain_mapping[var_id][1] != 0 {
                        domain_mapping[var_id][1] = 0;
                        changed = true;
                    }
                }
                if domain_mapping[var_id].len() >= 3 {
                    if domain_mapping[var_id][2] != 0 {
                        domain_mapping[var_id][2] = 0;
                        changed = true;
                    }
                }
                let _ = old_size; // keep structure similar to numeric-fd; size tracking handled elsewhere
            } else {
                let abs_size = domain_sizes[var_id];
                // If we've already fully refined this variable, nothing to do.
                if abs_size >= concrete_size {
                    return Ok(false);
                }
                // Only refine if the value is still mapped to the default class (0).
                if domain_mapping[var_id].get(value).copied().unwrap_or(0) != 0 {
                    return Ok(false);
                }
                if !can_refine_propositional_variable(
                    domain_sizes,
                    numeric_domain_sizes,
                    var_id,
                    abs_size + 1,
                    config.max_abstraction_size,
                ) {
                    return Ok(false);
                }

                domain_mapping[var_id][value] = abs_size;
                domain_sizes[var_id] = abs_size + 1;
                changed = true;
            }

            // Optional dependent numeric refinements (currently produced only for comparison vars).
            if dependent_numeric_refinement != DependentNumericRefinement::None
                && !pf.dependent_numeric_flaws.is_empty()
            {
                let mut any_numeric_changed = false;
                let iter: Box<dyn Iterator<Item = &NumericFlaw>> =
                    match dependent_numeric_refinement {
                        DependentNumericRefinement::None => Box::new(std::iter::empty()),
                        DependentNumericRefinement::All => {
                            Box::new(pf.dependent_numeric_flaws.iter())
                        }
                        DependentNumericRefinement::One => {
                            Box::new(pf.dependent_numeric_flaws.iter())
                        }
                    };

                for dep in iter {
                    let num_id = dep.numeric_var_id;

                    if !can_refine_numeric_variable(
                        domain_sizes,
                        numeric_domain_sizes,
                        num_id,
                        config.max_abstraction_size,
                    ) {
                        continue;
                    }

                    if partitions.split_at(num_id, dep.value, dep.include_in_lower) {
                        if let Some(parts) = partitions.partitions(num_id) {
                            if let Some(slot) = numeric_domain_sizes.get_mut(num_id) {
                                *slot = parts.len();
                            }
                        }
                        any_numeric_changed = true;
                        if dependent_numeric_refinement == DependentNumericRefinement::One {
                            break;
                        }
                    }
                }
                return Ok(any_numeric_changed || changed);
            }

            Ok(changed)
        }
    }
}

fn goal_variable_values(task: &dyn AbstractNumericTask) -> Vec<(usize, usize)> {
    let num_goals = task.get_num_goals();
    let mut goals = Vec::with_capacity(num_goals);
    for goal_idx in 0..num_goals {
        let goal = task.get_goal_fact(goal_idx);
        goals.push((goal.var, goal.value));
    }
    goals
}

fn choose_random_domain_value(domain_size: usize) -> usize {
    if domain_size <= 1 {
        0
    } else {
        let mut order: Vec<usize> = (0..domain_size).collect();
        shuffle_indices(&mut order);
        order[0]
    }
}

fn compute_initial_split_mapping(
    task: &dyn AbstractNumericTask,
    config: &CegarConfig,
    var_id: usize,
    goal_value: Option<usize>,
) -> Option<(usize, Vec<usize>)> {
    let concrete_domain_size = task.get_variable_domain_size(var_id).unwrap_or(0);
    if concrete_domain_size == 0 {
        return None;
    }

    let initial_value = task
        .get_initial_propositional_state_values()
        .get(var_id)
        .copied()
        .unwrap_or(0);

    match config.init_split_method {
        InitSplitMethod::GoalValue => {
            let goal = goal_value?;
            let mut mapping = vec![0; concrete_domain_size];
            mapping[goal] = 1;
            Some((2, mapping))
        }
        InitSplitMethod::GoalValueOrRandomIfNonGoal => {
            let chosen =
                goal_value.unwrap_or_else(|| choose_random_domain_value(concrete_domain_size));
            let mut mapping = vec![0; concrete_domain_size];
            mapping[chosen] = 1;
            Some((2, mapping))
        }
        InitSplitMethod::InitValue => {
            let mut mapping = vec![0; concrete_domain_size];
            if initial_value < mapping.len() {
                mapping[initial_value] = 1;
            }
            Some((2, mapping))
        }
        InitSplitMethod::RandomValue => {
            let chosen = choose_random_domain_value(concrete_domain_size);
            let mut mapping = vec![0; concrete_domain_size];
            mapping[chosen] = 1;
            Some((2, mapping))
        }
        InitSplitMethod::RandomPartition => {
            let mut order: Vec<usize> = (0..concrete_domain_size).collect();
            shuffle_indices(&mut order);
            let max_partition = choose_random_domain_value(concrete_domain_size).max(1);
            let mut mapping = vec![0; concrete_domain_size];
            for (index, concrete_value) in order.into_iter().enumerate() {
                mapping[concrete_value] = index % (max_partition + 1);
            }
            let abstract_domain_size = mapping.iter().copied().max().unwrap_or(0) + 1;
            Some((abstract_domain_size, mapping))
        }
        InitSplitMethod::RandomBinaryPartitionSeparatingInitGoal => {
            let mut mapping: Vec<usize> = (0..concrete_domain_size)
                .map(|_| choose_random_domain_value(2))
                .collect();
            if let Some(goal) = goal_value {
                if initial_value != goal && initial_value < mapping.len() && goal < mapping.len() {
                    mapping[initial_value] = 0;
                    mapping[goal] = 1;
                }
            }
            if mapping.iter().all(|&value| value == mapping[0]) {
                None
            } else {
                Some((2, mapping))
            }
        }
        InitSplitMethod::Identity => {
            Some((concrete_domain_size, (0..concrete_domain_size).collect()))
        }
    }
}

fn apply_initial_goal_splits(
    task: &dyn AbstractNumericTask,
    config: &CegarConfig,
    domain_mapping: &mut DomainMapping,
    domain_sizes: &mut [usize],
    numeric_domain_sizes: &[usize],
) {
    let goal_values: HashMap<usize, usize> = goal_variable_values(task).into_iter().collect();
    let mut candidate_var_ids: Vec<usize> = config
        .init_split_var_ids
        .as_ref()
        .map(|var_ids| var_ids.iter().copied().collect())
        .unwrap_or_else(|| goal_values.keys().copied().collect());
    candidate_var_ids.sort_unstable();
    candidate_var_ids.dedup();

    for var_id in candidate_var_ids {
        let Some((new_domain_size, mapping)) =
            compute_initial_split_mapping(task, config, var_id, goal_values.get(&var_id).copied())
        else {
            continue;
        };
        if new_domain_size <= 1 {
            continue;
        }
        if !can_refine_propositional_variable(
            domain_sizes,
            numeric_domain_sizes,
            var_id,
            new_domain_size,
            config.max_abstraction_size,
        ) {
            continue;
        }
        if let Some(slot) = domain_mapping.get_mut(var_id) {
            *slot = mapping;
        }
        if let Some(slot) = domain_sizes.get_mut(var_id) {
            *slot = new_domain_size;
        }
    }
}

pub fn run_cegar(task: &dyn AbstractNumericTask, config: CegarConfig) -> Result<CegarOutcome> {
    ensure!(
        config.max_abstraction_size > 0,
        "max_abstraction_size must be > 0"
    );
    ensure!(config.max_iterations > 0, "max_iterations must be > 0");

    let start = Instant::now();
    let cegar = Cegar::new(config.clone())?;

    let (mut domain_mapping, mut domain_sizes) =
        trivial_domain_mapping_and_sizes(task).context("failed to build trivial domain mapping")?;

    let mut partitions = NumericPartitions::trivial(task);
    let mut numeric_domain_sizes: Vec<usize> = vec![1; task.numeric_variables().len()];

    apply_initial_goal_splits(
        task,
        &config,
        &mut domain_mapping,
        &mut domain_sizes,
        &numeric_domain_sizes,
    );

    let mut iteration: usize = 1;
    let mut last_step: Option<CegarStep> = None;

    while iteration <= config.max_iterations {
        if let Some(max_time) = config.max_time {
            if start.elapsed() >= max_time {
                break;
            }
        }

        if config.debug {
            super::utils::debug_print_abstraction_stats(
                iteration,
                &domain_sizes,
                &numeric_domain_sizes,
            );
        }

        // TODO: avoid cloning at all cost.
        let factory = DomainAbstractionFactory::new(
            task,
            domain_mapping.clone(),
            domain_sizes.clone(),
            partitions.clone(),
            numeric_domain_sizes.clone(),
        )
        .with_context(|| {
            format!("failed to construct DomainAbstractionFactory (iteration {iteration})")
        })?;

        let wildcard_plan = if config.use_wildcard_plans {
            factory
                .compute_wildcard_plan(task, config.combine_labels, config.debug)
                .with_context(|| {
                    format!("failed to compute wildcard plan (iteration {iteration})")
                })?
        } else {
            let _table = factory
                .build_abstract_distance_table(task, config.combine_labels, false)
                .with_context(|| {
                    format!("failed to build abstract distance table (iteration {iteration})")
                })?;
            None
        };
        if config.debug {
            match wildcard_plan.as_ref() {
                Some(plan) => super::utils::debug_print_wildcard_plan(
                    task,
                    plan,
                    &domain_sizes,
                    &numeric_domain_sizes,
                    &partitions,
                ),
                None => println!("[Abstract Plan] <none>"),
            }
        }

        let step = CegarStep {
            factory,
            wildcard_plan,
        };
        last_step = Some(step);

        // Refinement requires a wildcard plan (current Rust port mirrors the numeric-fd flow).
        let Some(plan) = last_step.as_ref().and_then(|s| s.wildcard_plan.as_ref()) else {
            break;
        };

        let execute_entire_plan = match config.exec_entire_plan {
            ExecEntirePlanMode::StopAtFirstFlaw => false,
            ExecEntirePlanMode::ExecuteEntirePlan => true,
        };

        let flaws = cegar
            .get_flaws(task, &partitions, plan, execute_entire_plan)
            .with_context(|| format!("failed to collect flaws (iteration {iteration})"))?;
        if config.debug {
            super::utils::debug_print_flaws(&flaws);
        }
        if flaws.is_empty() {
            break;
        }

        let before_size = if config.debug {
            super::utils::compute_abstraction_size_u128(&domain_sizes, &numeric_domain_sizes)
        } else {
            None
        };
        let refined = cegar
            .fix_flaws(
                task,
                &flaws,
                &mut domain_mapping,
                &mut domain_sizes,
                &mut partitions,
                &mut numeric_domain_sizes,
            )
            .with_context(|| format!("failed to fix flaws (iteration {iteration})"))?;
        if config.debug {
            let after_size =
                super::utils::compute_abstraction_size_u128(&domain_sizes, &numeric_domain_sizes);
            super::utils::debug_print_refinement_summary(
                before_size,
                after_size,
                &domain_sizes,
                &numeric_domain_sizes,
                refined,
            );
        }
        if !refined {
            break;
        }

        iteration += 1;
    }

    let last_step = last_step.context("CEGAR did not perform any iterations")?;
    Ok(CegarOutcome {
        final_state: CegarState {
            domain_mapping,
            domain_sizes,
            partitions,
            numeric_domain_sizes,
            iteration,
        },
        last_step,
    })
}

fn trivial_domain_mapping_and_sizes(
    task: &dyn AbstractNumericTask,
) -> Result<(DomainMapping, Vec<usize>)> {
    let num_vars = task.get_num_variables();

    let mut domain_sizes: Vec<usize> = vec![1; num_vars];
    let mut domain_mapping: DomainMapping = Vec::with_capacity(num_vars);

    for var in 0..num_vars {
        let size = task
            .get_variable_domain_size(var)
            .map_err(|e| anyhow::anyhow!(e.to_string()))
            .with_context(|| format!("get_variable_domain_size({var}) failed"))?;
        ensure!(size > 0, "non-positive domain size for var {var}: {size}");
        domain_mapping.push(vec![0; size]);
    }

    Ok((domain_mapping, domain_sizes))
}

fn identity_domain_mapping_and_sizes(
    task: &dyn AbstractNumericTask,
) -> Result<(DomainMapping, Vec<usize>)> {
    let num_vars = task.get_num_variables();
    let mut domain_sizes: Vec<usize> = Vec::with_capacity(num_vars);
    let mut domain_mapping: DomainMapping = Vec::with_capacity(num_vars);

    for var in 0..num_vars {
        let size = task
            .get_variable_domain_size(var)
            .map_err(|e| anyhow::anyhow!(e.to_string()))
            .with_context(|| format!("get_variable_domain_size({var}) failed"))?;
        domain_sizes.push(size);

        let mut mapping: Vec<usize> = Vec::with_capacity(size);
        for val in 0..size {
            mapping.push(val);
        }
        domain_mapping.push(mapping);
    }

    Ok((domain_mapping, domain_sizes))
}

fn make_prop_state_packer(task: &dyn AbstractNumericTask) -> IntDoublePacker {
    let mut domain_sizes: Vec<u64> = Vec::with_capacity(task.variables().len());
    for var in task.variables().iter() {
        domain_sizes.push(var.domain_size() as u64);
    }
    IntDoublePacker::new(&domain_sizes)
}

fn set_initial_prop_values(
    task: &dyn AbstractNumericTask,
    packer: &IntDoublePacker,
    buffer: &mut [u64],
) {
    let init = task.get_initial_propositional_state_values();
    for (var_id, &val) in init.iter().enumerate() {
        packer.set(buffer, var_id, val as u64);
    }
}

fn fact_is_true(fact: &ExplicitFact, packer: &IntDoublePacker, buffer: &[u64]) -> bool {
    let current = packer.get(buffer, fact.var) as usize;
    current == fact.value
}

fn comparison_eval_code(v: Option<bool>) -> usize {
    match v {
        Some(true) => 0,
        Some(false) => 1,
        None => 2,
    }
}

fn determine_include_in_lower(
    tree: &super::comparison_expression::ComparisonTree,
    split_var_id: usize,
    split_value: f64,
    concrete_values: &[f64],
) -> bool {
    let mut lower_inputs: Vec<Interval> = concrete_values
        .iter()
        .copied()
        .map(Interval::singleton)
        .collect();
    let mut upper_inputs = lower_inputs.clone();

    if split_var_id < lower_inputs.len() {
        // If the split point is included in the lower interval, the current concrete
        // value belongs to (-inf, split_value].
        lower_inputs[split_var_id] = Interval::new(f64::NEG_INFINITY, split_value, false, true);
    }
    if split_var_id < upper_inputs.len() {
        // If the split point is included in the upper interval, the current concrete
        // value belongs to [split_value, inf).
        upper_inputs[split_var_id] = Interval::new(split_value, f64::INFINITY, true, false);
    }

    let eval_lower = comparison_eval_code(tree.evaluate_interval(&lower_inputs));
    let eval_upper = comparison_eval_code(tree.evaluate_interval(&upper_inputs));

    // Mirrors numeric-fd's preference: FALSE (=1) over UNKNOWN (=2) over TRUE (=0).
    if eval_lower == 1 && eval_upper != 1 {
        true
    } else if eval_upper == 1 && eval_lower != 1 {
        false
    } else if eval_lower == 1 && eval_upper == 1 {
        false
    } else if eval_lower == 2 && eval_upper == 2 {
        false
    } else if eval_lower == 2 {
        true
    } else if eval_upper == 2 {
        false
    } else {
        false
    }
}

fn dependent_numeric_flaws_for_comparison_prop_var(
    task: &dyn AbstractNumericTask,
    partitions: &NumericPartitions,
    comparison_index: &ComparisonAxiomIndex,
    prop_var_id: usize,
    numeric_state: &[f64],
) -> Vec<NumericFlaw> {
    let Some(tree) = comparison_index.comparison_tree(prop_var_id) else {
        return vec![];
    };

    let mut out: Vec<NumericFlaw> = Vec::new();
    for dep_var_id in tree.regular_numeric_var_dependencies(task) {
        let Some(&concrete_value) = numeric_state.get(dep_var_id) else {
            continue;
        };
        let include_in_lower =
            determine_include_in_lower(tree, dep_var_id, concrete_value, numeric_state);

        if can_split_numeric_var(partitions, dep_var_id, concrete_value, include_in_lower) {
            out.push(NumericFlaw {
                numeric_var_id: dep_var_id,
                value: concrete_value,
                include_in_lower,
            });
        } else if can_split_numeric_var(partitions, dep_var_id, concrete_value, !include_in_lower) {
            out.push(NumericFlaw {
                numeric_var_id: dep_var_id,
                value: concrete_value,
                include_in_lower: !include_in_lower,
            });
        }
    }
    out
}

pub fn get_precondition_flaws(
    task: &dyn AbstractNumericTask,
    partitions: &NumericPartitions,
    comparison_index: &ComparisonAxiomIndex,
    op: &planners_sas::numeric::numeric_task::Operator,
    packer: &IntDoublePacker,
    buffer: &[u64],
    numeric_state: &[f64],
) -> Vec<Flaw> {
    let mut out: Vec<Flaw> = Vec::new();
    for pre in op.preconditions().iter() {
        if !fact_is_true(pre, packer, buffer) {
            let prop_var_id = pre.var;
            let dependent_numeric_flaws =
                if comparison_index.is_comparison_axiom_variable(prop_var_id) {
                    dependent_numeric_flaws_for_comparison_prop_var(
                        task,
                        partitions,
                        comparison_index,
                        prop_var_id,
                        numeric_state,
                    )
                } else {
                    vec![]
                };
            out.push(Flaw::Propositional(PropFlaw {
                fact: pre.clone(),
                dependent_numeric_flaws,
            }));
        }
    }
    out
}

fn get_goal_flaws(
    task: &dyn AbstractNumericTask,
    partitions: &NumericPartitions,
    comparison_index: &ComparisonAxiomIndex,
    packer: &IntDoublePacker,
    buffer: &[u64],
    numeric_state: &[f64],
) -> Vec<Flaw> {
    let num_goals_i32 = task.get_num_goals();
    let num_goals = usize::try_from(num_goals_i32.max(0)).unwrap_or(0);
    let mut out: Vec<Flaw> = Vec::new();
    let mut seen: BTreeSet<ExplicitFact> = BTreeSet::new();
    let mut derived_goal_vars: BTreeSet<usize> = BTreeSet::new();
    for goal_id in 0..num_goals {
        let goal_fact = task.get_goal_fact(goal_id);
        let goal_var = goal_fact.var;
        let goal_is_derived = task.axioms().iter().any(|ax| ax.var_id() == goal_var);
        if goal_is_derived {
            derived_goal_vars.insert(goal_var);
            continue;
        }
        if !fact_is_true(goal_fact, packer, buffer) && seen.insert(goal_fact.clone()) {
            let prop_var_id = goal_fact.var;
            let dependent_numeric_flaws =
                if comparison_index.is_comparison_axiom_variable(prop_var_id) {
                    dependent_numeric_flaws_for_comparison_prop_var(
                        task,
                        partitions,
                        comparison_index,
                        prop_var_id,
                        numeric_state,
                    )
                } else {
                    vec![]
                };
            out.push(Flaw::Propositional(PropFlaw {
                fact: goal_fact.clone(),
                dependent_numeric_flaws,
            }));
        }
    }

    // Reconstruct (potentially hidden) goal conditions from propositional goal axioms.
    for ax in task.axioms().iter() {
        if ax.conditions().is_empty() {
            continue;
        }
        if !derived_goal_vars.is_empty() && !derived_goal_vars.contains(&ax.var_id()) {
            continue;
        }
        for pre in ax.conditions().iter() {
            if !fact_is_true(pre, packer, buffer) && seen.insert(pre.clone()) {
                let prop_var_id = pre.var;
                let dependent_numeric_flaws =
                    if comparison_index.is_comparison_axiom_variable(prop_var_id) {
                        dependent_numeric_flaws_for_comparison_prop_var(
                            task,
                            partitions,
                            comparison_index,
                            prop_var_id,
                            numeric_state,
                        )
                    } else {
                        vec![]
                    };
                out.push(Flaw::Propositional(PropFlaw {
                    fact: pre.clone(),
                    dependent_numeric_flaws,
                }));
            }
        }
    }
    out
}

pub(crate) fn apply_operator_to_state(
    op: &planners_sas::numeric::numeric_task::Operator,
    packer: &IntDoublePacker,
    buffer: &mut [u64],
    numeric_state: &mut Vec<f64>,
) {
    // Propositional effects (respect conditions).
    for eff in op.effects().iter() {
        let mut ok = true;
        for cond in eff.conditions().iter() {
            if !fact_is_true(cond, packer, buffer) {
                ok = false;
                break;
            }
        }
        if ok {
            packer.set(buffer, eff.var_id(), eff.value() as u64);
        }
    }

    // Numeric assignment effects.
    for eff in op.assignment_effects().iter() {
        if eff.is_conditional() {
            let mut ok = true;
            for cond in eff.conditions().iter() {
                if !fact_is_true(cond, packer, buffer) {
                    ok = false;
                    break;
                }
            }
            if !ok {
                continue;
            }
        }

        let assignment_var_id = eff.var_id() as usize;
        let affected_var_id = eff.affected_var_id() as usize;
        if assignment_var_id >= numeric_state.len() || affected_var_id >= numeric_state.len() {
            continue;
        }
        let operand = numeric_state[assignment_var_id];
        numeric_state[affected_var_id] =
            planners_sas::numeric::numeric_task::AssignmentOperation::apply(
                numeric_state[affected_var_id],
                eff.operation(),
                operand,
            );
    }
}

fn partition_for_value(
    partitions: &[super::comparison_expression::Interval],
    value: f64,
) -> Option<usize> {
    partitions.iter().position(|iv| iv.contains(value))
}

fn can_split_numeric_var(
    partitions: &NumericPartitions,
    numeric_var_id: usize,
    value: f64,
    include_in_lower: bool,
) -> bool {
    let Some(parts) = partitions.partitions(numeric_var_id) else {
        return false;
    };
    let Some(part_id) = parts.iter().position(|iv| iv.contains(value)) else {
        return false;
    };
    parts[part_id].can_split_at(value, include_in_lower)
}

pub fn get_numeric_deviation_flaws(
    op: &planners_sas::numeric::numeric_task::Operator,
    numeric_current_state: &[f64],
    numeric_successor_state: &[f64],
    abstract_numeric_successor_state: &[usize],
    partitions: &NumericPartitions,
) -> Vec<Flaw> {
    let mut flaws: Vec<Flaw> = Vec::new();

    let num_vars = numeric_successor_state
        .len()
        .min(abstract_numeric_successor_state.len());
    for var_id in 0..num_vars {
        let operator_modified_var = op
            .assignment_effects()
            .iter()
            .any(|eff| eff.affected_var_id() == var_id);
        if !operator_modified_var {
            continue;
        }

        let abstract_value = abstract_numeric_successor_state[var_id];
        let Some(parts) = partitions.partitions(var_id) else {
            continue;
        };
        let Some(correct_abstract_value) =
            partition_for_value(parts, numeric_successor_state[var_id])
        else {
            continue;
        };
        if abstract_value == correct_abstract_value {
            continue;
        }

        let concrete_next_value = numeric_successor_state[var_id];
        let concrete_current_value = numeric_current_state
            .get(var_id)
            .copied()
            .unwrap_or(concrete_next_value);
        if concrete_next_value == concrete_current_value {
            continue;
        }

        let operator_increased_value = concrete_next_value > concrete_current_value;
        let mut include_in_lower = !operator_increased_value;

        if can_split_numeric_var(partitions, var_id, concrete_current_value, include_in_lower) {
            flaws.push(Flaw::Numeric(NumericFlaw {
                numeric_var_id: var_id,
                value: concrete_current_value,
                include_in_lower,
            }));
        } else {
            include_in_lower = !include_in_lower;
            if can_split_numeric_var(partitions, var_id, concrete_current_value, include_in_lower) {
                flaws.push(Flaw::Numeric(NumericFlaw {
                    numeric_var_id: var_id,
                    value: concrete_current_value,
                    include_in_lower,
                }));
            }
        }
    }

    flaws
}
