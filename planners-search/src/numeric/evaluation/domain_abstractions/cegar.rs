#[cfg(test)]
mod tests;

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt::Write as _;
use std::time::{Duration, Instant};

use anyhow::{ensure, Context, Result};

use libc::exit;
use planners_sas::numeric::axioms::AxiomEvaluator;
use planners_sas::numeric::numeric_task::{AbstractNumericTask, Fact};
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
	pub fact: Fact,
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlawTreatment {
	RandomSingleAtom,
	OneSplitPerAtom,
	OneSplitPerVariable,
	MaxRefinedSingleAtom,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DependentNumericRefinement {
	None,
	One,
	All,
}

#[derive(Debug, Clone)]
pub struct CegarConfig {
	pub max_iterations: usize,
	pub max_time: Option<Duration>,
	pub use_wildcard_plans: bool,
	pub combine_labels: bool,
	pub enable_refinement: bool,
	pub debug: bool,
	pub refinement_batch_size: usize,
	pub flaw_treatment: FlawTreatment,
	pub use_progress_weighted_flaw_selection: bool,
}

impl Default for CegarConfig {
	fn default() -> Self {
		Self {
			max_iterations: 10_000,
			max_time: None,
			use_wildcard_plans: true,
			combine_labels: true,
			enable_refinement: false,
			debug: false,
			refinement_batch_size: 1,
			flaw_treatment: FlawTreatment::RandomSingleAtom,
			use_progress_weighted_flaw_selection: false,
		}
	}
}

#[derive(Debug, Clone)]
pub struct CegarState {
	pub domain_mapping: DomainMapping,
	pub domain_sizes: Vec<i32>,
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
		ensure!(config.max_iterations > 0, "max_iterations must be > 0");
		ensure!(config.refinement_batch_size > 0, "refinement_batch_size must be > 0");
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
		let comparison_index =
			ComparisonAxiomIndex::from_task(task).map_err(|e| anyhow::anyhow!("failed to build ComparisonAxiomIndex: {e}"))?;

		let state_packer = make_prop_state_packer(task);
		let axiom_evaluator = AxiomEvaluator::new(task, &state_packer);

		let mut buffer = vec![0u64; state_packer.num_bins() as usize];
		set_initial_prop_values(task, &state_packer, &mut buffer);
		let mut numeric_state: Vec<f64> = task.get_initial_numeric_state_values().to_vec();

		axiom_evaluator
			.evaluate_arithmetic_axioms(&mut numeric_state)
			.map_err(|e| anyhow::anyhow!("failed to evaluate arithmetic axioms for initial state: {e:?}"))?;
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
							.map_err(|e| anyhow::anyhow!("failed to evaluate arithmetic axioms after operator: {e:?}"))?;
						axiom_evaluator
							.evaluate(&mut candidate_buffer, &mut candidate_numeric_state)
							.map_err(|e| anyhow::anyhow!("failed to evaluate axioms after operator: {e:?}"))?;

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
					step_flaws.clear();
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
					.map_err(|e| anyhow::anyhow!("failed to evaluate arithmetic axioms after operator: {e:?}"))?;
				axiom_evaluator
					.evaluate(&mut buffer, &mut numeric_state)
					.map_err(|e| anyhow::anyhow!("failed to evaluate axioms after operator: {e:?}"))?;

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
		domain_sizes: &mut Vec<i32>,
		partitions: &mut NumericPartitions,
		numeric_domain_sizes: &mut Vec<usize>,
	) -> bool {
		let comparison_var_ids: HashSet<usize> = task
			.comparison_axioms()
			.iter()
			.filter_map(|ax| usize::try_from(ax.get_affected_var_id()).ok())
			.collect();
		let abstraction_size = compute_abstraction_size(domain_sizes, numeric_domain_sizes);

		if self.config.refinement_batch_size > 1 {
			return fix_top_k_flaws(
				task,
				flaws,
				&comparison_var_ids,
				domain_mapping,
				domain_sizes,
				partitions,
				numeric_domain_sizes,
				abstraction_size,
				self.config.refinement_batch_size,
				self.config.use_progress_weighted_flaw_selection,
			);
		}

		match self.config.flaw_treatment {
			FlawTreatment::RandomSingleAtom => fix_single_flaw_in_order(
				task,
				flaws,
				&comparison_var_ids,
				domain_mapping,
				domain_sizes,
				partitions,
				numeric_domain_sizes,
				abstraction_size,
				self.config.use_progress_weighted_flaw_selection,
			),
			FlawTreatment::OneSplitPerAtom => fix_flaws_per_atom(
				task,
				flaws,
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

fn compute_abstraction_size(domain_sizes: &[i32], numeric_domain_sizes: &[usize]) -> usize {
	let mut size: usize = 1;
	for &d in domain_sizes.iter() {
		let d_usize = usize::try_from(d.max(0)).unwrap_or(0);
		if d_usize == 0 {
			return 0;
		}
		size = size.saturating_mul(d_usize);
	}
	for &p in numeric_domain_sizes.iter() {
		if p == 0 {
			return 0;
		}
		size = size.saturating_mul(p);
	}
	size
}

fn flaw_atom_key(flaw: &Flaw) -> (u8, usize, usize, u64, bool) {
	match flaw {
		Flaw::Propositional(pf) => (0, pf.fact.var() as usize, pf.fact.value() as usize, 0, false),
		Flaw::Numeric(nf) => (1, nf.numeric_var_id, 0, nf.value.to_bits(), nf.include_in_lower),
	}
}

fn flaw_variable_key(flaw: &Flaw) -> (u8, usize) {
	match flaw {
		Flaw::Propositional(pf) => (0, pf.fact.var() as usize),
		Flaw::Numeric(nf) => (1, nf.numeric_var_id),
	}
}

fn score_flaw(flaw: &Flaw, domain_sizes: &[i32], numeric_domain_sizes: &[usize], _abstraction_size: usize) -> i64 {
	match flaw {
		Flaw::Numeric(nf) => numeric_domain_sizes
			.get(nf.numeric_var_id)
			.copied()
			.unwrap_or(0) as i64,
		Flaw::Propositional(pf) => {
			let var_id = pf.fact.var() as usize;
			let base = domain_sizes.get(var_id).copied().unwrap_or(0) as i64;
			let max_dep = pf
				.dependent_numeric_flaws
				.iter()
				.filter_map(|nf| numeric_domain_sizes.get(nf.numeric_var_id).copied())
				.max()
				.unwrap_or(0) as i64;
			base + max_dep
		}
	}
}

fn fix_top_k_flaws(
	task: &dyn AbstractNumericTask,
	flaws: &[Flaw],
	comparison_var_ids: &HashSet<usize>,
	domain_mapping: &mut DomainMapping,
	domain_sizes: &mut Vec<i32>,
	partitions: &mut NumericPartitions,
	numeric_domain_sizes: &mut Vec<usize>,
	abstraction_size: usize,
	refinement_batch_size: usize,
	use_progress_weighted_flaw_selection: bool,
) -> bool {
	if flaws.is_empty() {
		return false;
	}

	let mut indices: Vec<usize> = (0..flaws.len()).collect();
	if use_progress_weighted_flaw_selection {
		indices.sort_by(|&a, &b| {
			let sa = score_flaw(&flaws[a], domain_sizes, numeric_domain_sizes, abstraction_size);
			let sb = score_flaw(&flaws[b], domain_sizes, numeric_domain_sizes, abstraction_size);
			sb.cmp(&sa).then_with(|| flaw_atom_key(&flaws[a]).cmp(&flaw_atom_key(&flaws[b])))
		});
	}

	let mut changed = false;
	let mut applied: usize = 0;
	let mut refined_prop_vars: HashSet<usize> = HashSet::new();
	let mut refined_numeric_vars: HashSet<usize> = HashSet::new();

	for idx in indices {
		if applied >= refinement_batch_size {
			break;
		}
		let local_changed = try_refine_from_flaw(
			task,
			&flaws[idx],
			comparison_var_ids,
			domain_mapping,
			domain_sizes,
			partitions,
			numeric_domain_sizes,
			DependentNumericRefinement::One,
			Some(&mut refined_prop_vars),
			Some(&mut refined_numeric_vars),
		);
		if local_changed {
			changed = true;
			applied += 1;
		}
	}

	changed
}

fn fix_single_flaw_in_order(
	task: &dyn AbstractNumericTask,
	flaws: &[Flaw],
	comparison_var_ids: &HashSet<usize>,
	domain_mapping: &mut DomainMapping,
	domain_sizes: &mut Vec<i32>,
	partitions: &mut NumericPartitions,
	numeric_domain_sizes: &mut Vec<usize>,
	abstraction_size: usize,
	use_progress_weighted_flaw_selection: bool,
) -> bool {
	if flaws.is_empty() {
		return false;
	}

	let mut indices: Vec<usize> = (0..flaws.len()).collect();
	if use_progress_weighted_flaw_selection {
		indices.sort_by(|&a, &b| {
			let sa = score_flaw(&flaws[a], domain_sizes, numeric_domain_sizes, abstraction_size);
			let sb = score_flaw(&flaws[b], domain_sizes, numeric_domain_sizes, abstraction_size);
			sb.cmp(&sa).then_with(|| flaw_atom_key(&flaws[a]).cmp(&flaw_atom_key(&flaws[b])))
		});
	}

	for idx in indices {
		if try_refine_from_flaw(
			task,
			&flaws[idx],
			comparison_var_ids,
			domain_mapping,
			domain_sizes,
			partitions,
			numeric_domain_sizes,
			DependentNumericRefinement::One,
			None,
			None,
		) {
			return true;
		}
	}

	false
}

fn fix_flaws_per_atom(
	task: &dyn AbstractNumericTask,
	flaws: &[Flaw],
	comparison_var_ids: &HashSet<usize>,
	domain_mapping: &mut DomainMapping,
	domain_sizes: &mut Vec<i32>,
	partitions: &mut NumericPartitions,
	numeric_domain_sizes: &mut Vec<usize>,
	_abstraction_size: usize,
) -> bool {
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
			comparison_var_ids,
			domain_mapping,
			domain_sizes,
			partitions,
			numeric_domain_sizes,
			DependentNumericRefinement::All,
			None,
			None,
		);
		changed = changed || local_changed;
	}
	changed
}

fn fix_flaws_per_variable(
	task: &dyn AbstractNumericTask,
	flaws: &[Flaw],
	comparison_var_ids: &HashSet<usize>,
	domain_mapping: &mut DomainMapping,
	domain_sizes: &mut Vec<i32>,
	partitions: &mut NumericPartitions,
	numeric_domain_sizes: &mut Vec<usize>,
	_abstraction_size: usize,
) -> bool {
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
			comparison_var_ids,
			domain_mapping,
			domain_sizes,
			partitions,
			numeric_domain_sizes,
			DependentNumericRefinement::One,
			Some(&mut refined_prop_vars),
			Some(&mut refined_numeric_vars),
		);
		changed = changed || local_changed;
	}
	changed
}

fn fix_single_flaw_max_refined(
	task: &dyn AbstractNumericTask,
	flaws: &[Flaw],
	comparison_var_ids: &HashSet<usize>,
	domain_mapping: &mut DomainMapping,
	domain_sizes: &mut Vec<i32>,
	partitions: &mut NumericPartitions,
	numeric_domain_sizes: &mut Vec<usize>,
	abstraction_size: usize,
) -> bool {
	if flaws.is_empty() {
		return false;
	}

	#[derive(Clone)]
	struct Candidate {
		idx: usize,
		score: i64,
		restricted_dep: Option<Vec<NumericFlaw>>,
	}

	let mut candidates: Vec<Candidate> = Vec::with_capacity(flaws.len());
	for (idx, flaw) in flaws.iter().enumerate() {
		let mut restricted_dep: Option<Vec<NumericFlaw>> = None;
		let score = match flaw {
			Flaw::Numeric(nf) => numeric_domain_sizes.get(nf.numeric_var_id).copied().unwrap_or(0) as i64,
			Flaw::Propositional(pf) => {
				let var_id = pf.fact.var() as usize;
				let base = domain_sizes.get(var_id).copied().unwrap_or(0) as i64;
				if comparison_var_ids.contains(&var_id) && !pf.dependent_numeric_flaws.is_empty() {
					let mut best: BTreeMap<usize, Vec<NumericFlaw>> = BTreeMap::new();
					for nf in pf.dependent_numeric_flaws.iter().cloned() {
						let partitions = numeric_domain_sizes.get(nf.numeric_var_id).copied().unwrap_or(0);
						best.entry(partitions).or_default().push(nf);
					}
					if let Some((&max_partitions, vec)) = best.iter().next_back() {
						restricted_dep = Some(vec.clone());
						base + (max_partitions as i64)
					} else {
						base
					}
				} else {
					base
				}
			}
		};
		candidates.push(Candidate { idx, score, restricted_dep });
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
			comparison_var_ids,
			domain_mapping,
			domain_sizes,
			partitions,
			numeric_domain_sizes,
			DependentNumericRefinement::One,
			None,
			None,
		) {
			return true;
		}
	}

	let _ = abstraction_size;
	false
}

fn try_refine_from_flaw(
	task: &dyn AbstractNumericTask,
	flaw: &Flaw,
	comparison_var_ids: &HashSet<usize>,
	domain_mapping: &mut DomainMapping,
	domain_sizes: &mut [i32],
	partitions: &mut NumericPartitions,
	numeric_domain_sizes: &mut [usize],
	dependent_numeric_refinement: DependentNumericRefinement,
	mut refined_prop_vars: Option<&mut HashSet<usize>>,
	mut refined_numeric_vars: Option<&mut HashSet<usize>>,
) -> bool {
	match flaw {
		Flaw::Numeric(nf) => {
			let var_id = nf.numeric_var_id;
			if let Some(set) = refined_numeric_vars.as_ref() {
				if set.contains(&var_id) {
					return false;
				}
			}
			if partitions.split_at(var_id, nf.value, nf.include_in_lower) {
				if let Some(parts) = partitions.partitions(var_id) {
					if let Some(slot) = numeric_domain_sizes.get_mut(var_id) {
						*slot = parts.len();
					}
				}
				if let Some(set) = refined_numeric_vars.as_deref_mut() {
					set.insert(var_id);
				}
				return true;
			}
			false
		}
		Flaw::Propositional(pf) => {
			let var_id = pf.fact.var() as usize;
			let value = pf.fact.value() as usize;
			if let Some(set) = refined_prop_vars.as_ref() {
				if set.contains(&var_id) {
					return false;
				}
			}
			if var_id >= domain_mapping.len() || var_id >= domain_sizes.len() {
				return false;
			}

			let Ok(var_i32) = i32::try_from(var_id) else {
				return false;
			};
			let Ok(concrete_size) = task.get_variable_domain_size(var_i32) else {
				return false;
			};
			let Ok(concrete_size_usize) = usize::try_from(concrete_size.max(0)) else {
				return false;
			};
			if value >= concrete_size_usize {
				return false;
			}

			let mut changed = false;

			if comparison_var_ids.contains(&var_id) {
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
				if abs_size <= 0 {
					return false;
				}
				let Ok(abs_size_usize) = usize::try_from(abs_size) else {
					return false;
				};
				// If we've already fully refined this variable, nothing to do.
				if abs_size_usize >= concrete_size_usize {
					return false;
				}
				// Only refine if the value is still mapped to the default class (0).
				if domain_mapping[var_id].get(value).copied().unwrap_or(0) != 0 {
					return false;
				}

				domain_mapping[var_id][value] = abs_size;
				domain_sizes[var_id] = abs_size + 1;
				changed = true;
			}

			if let Some(set) = refined_prop_vars.as_deref_mut() {
				set.insert(var_id);
			}

			// Optional dependent numeric refinements (currently produced only for comparison vars).
			if dependent_numeric_refinement != DependentNumericRefinement::None
				&& !pf.dependent_numeric_flaws.is_empty()
			{
				let mut any_numeric_changed = false;
				let iter: Box<dyn Iterator<Item = &NumericFlaw>> = match dependent_numeric_refinement {
					DependentNumericRefinement::None => Box::new(std::iter::empty()),
					DependentNumericRefinement::All => Box::new(pf.dependent_numeric_flaws.iter()),
					DependentNumericRefinement::One => Box::new(pf.dependent_numeric_flaws.iter()),
				};

				for dep in iter {
					let num_id = dep.numeric_var_id;
					if let Some(set) = refined_numeric_vars.as_ref() {
						if set.contains(&num_id) {
							continue;
						}
					}

					if partitions.split_at(num_id, dep.value, dep.include_in_lower) {
						if let Some(parts) = partitions.partitions(num_id) {
							if let Some(slot) = numeric_domain_sizes.get_mut(num_id) {
								*slot = parts.len();
							}
						}
						if let Some(set) = refined_numeric_vars.as_deref_mut() {
							set.insert(num_id);
						}
						any_numeric_changed = true;
						if dependent_numeric_refinement == DependentNumericRefinement::One {
							break;
						}
					}
				}
				return any_numeric_changed || changed;
			}

			changed
		}
	}
}

pub fn run_cegar(task: &dyn AbstractNumericTask, config: CegarConfig) -> Result<CegarOutcome> {
	ensure!(config.max_iterations > 0, "max_iterations must be > 0");

	let start = Instant::now();
	let cegar = Cegar::new(config.clone())?;

	let (mut domain_mapping, mut domain_sizes) = trivial_domain_mapping_and_sizes(task)
		.context("failed to build trivial domain mapping")?;

	let mut partitions = NumericPartitions::trivial(task);
	let mut numeric_domain_sizes: Vec<usize> = vec![1; task.numeric_variables().len()];

	// TODO: initialization split strategies (init/goal/random/identity/etc).
	// This is where we would apply initial splits and update `domain_mapping`, `domain_sizes`,
	// `partitions`, and `numeric_domain_sizes` before the first abstraction is built.

	let mut iteration: usize = 1;
	let mut last_step: Option<CegarStep> = None;

	while iteration <= config.max_iterations {
		if let Some(max_time) = config.max_time {
			if start.elapsed() >= max_time {
				break;
			}
		}

		if config.debug {
			debug_print_abstraction_stats(iteration, &domain_sizes, &numeric_domain_sizes);
		}

        // TODO: avoid cloning at all cost. 
		let factory = DomainAbstractionFactory::new(
			task,
			domain_mapping.clone(),
			domain_sizes.clone(),
			partitions.clone(),
			numeric_domain_sizes.clone(),
		)
		.with_context(|| format!("failed to construct DomainAbstractionFactory (iteration {iteration})"))?;

		let wildcard_plan = if config.use_wildcard_plans {
			factory
				.compute_wildcard_plan(task, config.combine_labels, config.debug)
				.with_context(|| format!("failed to compute wildcard plan (iteration {iteration})"))?
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
				Some(plan) => debug_print_wildcard_plan(task, plan, &domain_sizes, &numeric_domain_sizes, &partitions),
				None => println!("[Abstract Plan] <none>"),
			}
		}

		let step = CegarStep {
			factory,
			wildcard_plan,
		};
		last_step = Some(step);

		if !config.enable_refinement {
			break;
		}

		// Refinement requires a wildcard plan (current Rust port mirrors the numeric-fd flow).
		let Some(plan) = last_step
			.as_ref()
			.and_then(|s| s.wildcard_plan.as_ref())
		else {
			break;
		};

		let flaws = cegar
			.get_flaws(task, &partitions, plan, false)
			.with_context(|| format!("failed to collect flaws (iteration {iteration})"))?;
		if config.debug {
			debug_print_flaws(task, &flaws);
		}
		if flaws.is_empty() {
			break;
		}

		let before_size = if config.debug {
			compute_abstraction_size_u128(&domain_sizes, &numeric_domain_sizes)
		} else {
			None
		};
		let refined = cegar.fix_flaws(
			task,
			&flaws,
			&mut domain_mapping,
			&mut domain_sizes,
			&mut partitions,
			&mut numeric_domain_sizes,
		);
		if config.debug {
			let after_size = compute_abstraction_size_u128(&domain_sizes, &numeric_domain_sizes);
			debug_print_refinement_summary(before_size, after_size, &domain_sizes, &numeric_domain_sizes, refined);
		}
		if !refined {
			break;
		}

		iteration += 1;
		if iteration > 6 {
			unsafe {
				exit(0);
			}
		}
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
) -> Result<(DomainMapping, Vec<i32>)> {
	let num_vars_i32 = task.get_num_variables();
	ensure!(num_vars_i32 >= 0, "task.get_num_variables() must be non-negative");
	let num_vars = usize::try_from(num_vars_i32).context("num_vars does not fit usize")?;

	let mut domain_sizes: Vec<i32> = vec![1; num_vars];
	let mut domain_mapping: DomainMapping = Vec::with_capacity(num_vars);

	for var in 0..num_vars {
		let var_i32 = i32::try_from(var).context("var index does not fit i32")?;
		let size = task
			.get_variable_domain_size(var_i32)
			.map_err(|e| anyhow::anyhow!(e.to_string()))
			.with_context(|| format!("get_variable_domain_size({var}) failed"))?;
		ensure!(size > 0, "non-positive domain size for var {var}: {size}");
		domain_mapping.push(vec![0; size as usize]);
	}

	Ok((domain_mapping, domain_sizes))
}

fn compute_abstraction_size_u128(domain_sizes: &[i32], numeric_domain_sizes: &[usize]) -> Option<u128> {
	let mut size: u128 = 1;
	for &d in domain_sizes.iter() {
		let du = u128::try_from(d).ok()?;
		if du == 0 {
			return Some(0);
		}
		size = size.checked_mul(du)?;
	}
	for &p in numeric_domain_sizes.iter() {
		let pu = u128::try_from(p).ok()?;
		if pu == 0 {
			return Some(0);
		}
		size = size.checked_mul(pu)?;
	}
	Some(size)
}

fn debug_print_abstraction_stats(iteration: usize, domain_sizes: &[i32], numeric_domain_sizes: &[usize]) {
	let prop_vars = domain_sizes.len();
	let num_vars = numeric_domain_sizes.len();
	let refined_props = domain_sizes.iter().filter(|&&s| s > 1).count();
	let refined_nums = numeric_domain_sizes.iter().filter(|&&s| s > 1).count();
	let size = compute_abstraction_size_u128(domain_sizes, numeric_domain_sizes)
		.map(|s| s.to_string())
		.unwrap_or_else(|| "<overflow>".to_string());

	let prop_max = domain_sizes.iter().copied().max().unwrap_or(0);
	let num_max = numeric_domain_sizes.iter().copied().max().unwrap_or(0);

	println!(
		"[CEGAR] iteration {iteration}: abstract_states={size} (prop_vars={prop_vars}, num_vars={num_vars}, refined_prop={refined_props}, refined_num={refined_nums}, max_prop_size={prop_max}, max_num_parts={num_max})"
	);
}

fn debug_print_wildcard_plan(
	task: &dyn AbstractNumericTask,
	plan: &WildcardPlanResult,
	domain_sizes: &[i32],
	numeric_domain_sizes: &[usize],
	partitions: &NumericPartitions,
) {
	let steps = plan.wildcard_plan.len();
	println!("[Abstract Plan] steps={steps}");

	let max_steps = 200usize;
	let shown_steps = steps.min(max_steps);
	if steps > max_steps {
		println!("[Abstract Plan] (truncated to first {shown_steps} steps)");
	}

	// Print initial abstract state snapshot (non-trivial vars only; includes zeros).
	if let Some(prop0) = plan.abstract_prop_states.first() {
		println!("  s0 props: {}", fmt_nontrivial_props(prop0, domain_sizes, 100));
	}
	if let Some(num0) = plan.abstract_numeric_states.first() {
		println!(
			"  s0 nums:  {}",
			fmt_nontrivial_nums(num0, numeric_domain_sizes, partitions, 100)
		);
	}

	let ops = task.get_operators();
	let mut representative: Vec<String> = Vec::with_capacity(shown_steps);

	for i in 0..shown_steps {
		let choices = &plan.wildcard_plan[i];
		let choice_count = choices.len();
		let rep = choices
			.first()
			.and_then(|&id| ops.get(id).map(|op| format!("{}", op.name())))
			.unwrap_or_else(|| "<none>".to_string());
		representative.push(rep);

		let mut line = String::new();
		let _ = write!(&mut line, "  step {i}: options={choice_count}");
		let preview = 10usize;
		for &op_id in choices.iter().take(preview) {
			let name = ops.get(op_id).map(|op| op.name()).unwrap_or("<bad-op-id>");
			let _ = write!(&mut line, " [{op_id}:{name}]");
		}
		if choice_count > preview {
			let _ = write!(&mut line, " ...");
		}
		println!("{line}");

		// Print abstract state deltas if available.
		if i + 1 < plan.abstract_prop_states.len() {
			let prev = &plan.abstract_prop_states[i];
			let cur = &plan.abstract_prop_states[i + 1];
			let delta = fmt_delta_i32(prev, cur, 50);
			if !delta.is_empty() {
				println!("    props Δ: {delta}");
			}
		}
		if i + 1 < plan.abstract_numeric_states.len() {
			let prev = &plan.abstract_numeric_states[i];
			let cur = &plan.abstract_numeric_states[i + 1];
			let delta = fmt_delta_numeric_partitions(prev, cur, partitions, 50);
			if !delta.is_empty() {
				println!("    nums  Δ: {delta}");
			}
		}
	}

	println!("[Plan] {}", representative.join(" -> "));
	debug_print_concrete_trace(task, plan, partitions, shown_steps);
}

fn debug_print_concrete_trace(
	task: &dyn AbstractNumericTask,
	plan: &WildcardPlanResult,
	partitions: &NumericPartitions,
	shown_steps: usize,
) {
	let state_packer = make_prop_state_packer(task);
	let axiom_evaluator = AxiomEvaluator::new(task, &state_packer);

	let mut buffer = vec![0u64; state_packer.num_bins() as usize];
	set_initial_prop_values(task, &state_packer, &mut buffer);
	let mut numeric_state: Vec<f64> = task.get_initial_numeric_state_values().to_vec();

	let _ = axiom_evaluator.evaluate_arithmetic_axioms(&mut numeric_state);
	let _ = axiom_evaluator.evaluate(&mut buffer, &mut numeric_state);

	let (prop_scope, num_scope) = trace_variable_scope(task, plan, shown_steps);
	println!("[Concrete Trace] scope: props={} nums={}", prop_scope.len(), num_scope.len());
	println!(
		"  s0 props: {}",
		fmt_concrete_props(task, &state_packer, &buffer, &prop_scope, 200)
	);
	println!(
		"  s0 nums:  {}",
		fmt_concrete_nums(&numeric_state, &num_scope, partitions, 200)
	);

	let comparison_index = ComparisonAxiomIndex::from_task(task).ok();

	let max_tries_per_step = 30usize;
	for step in 0..shown_steps {
		if step + 1 >= plan.abstract_numeric_states.len() {
			break;
		}
		let expected_abs_numeric_succ = &plan.abstract_numeric_states[step + 1];
		let choices = plan.wildcard_plan.get(step).map(|v| v.as_slice()).unwrap_or(&[]);

		let mut chosen: Option<(usize, Vec<u64>, Vec<f64>)> = None;
		let mut tries = 0usize;
		for &op_id in choices.iter() {
			if tries >= max_tries_per_step {
				println!("  step {step}: ... (tried first {max_tries_per_step} options)");
				break;
			}
			let Some(op) = task.get_operators().get(op_id) else {
				continue;
			};
			tries += 1;

			let applicable = if let Some(idx) = comparison_index.as_ref() {
				get_precondition_flaws(task, partitions, idx, op, &state_packer, &buffer, &numeric_state)
					.is_empty()
			} else {
				op.preconditions().iter().all(|pre| fact_is_true(pre, &state_packer, &buffer))
			};
			if !applicable {
				continue;
			}

			let mut cand_buffer = buffer.clone();
			let mut cand_numeric = numeric_state.clone();
			apply_operator_to_state(op, &state_packer, &mut cand_buffer, &mut cand_numeric);
			let _ = axiom_evaluator.evaluate_arithmetic_axioms(&mut cand_numeric);
			let _ = axiom_evaluator.evaluate(&mut cand_buffer, &mut cand_numeric);

			let deviation_flaws = get_numeric_deviation_flaws(
				op,
				&numeric_state,
				&cand_numeric,
				expected_abs_numeric_succ,
				partitions,
			);

			if deviation_flaws.is_empty() {
				println!("  step {step}: choose [{op_id}:{}]", op.name());
				chosen = Some((op_id, cand_buffer, cand_numeric));
				break;
			} else {
				// We did encounter this successor state while testing a wildcard option.
				println!("  step {step}: try    [{op_id}:{}] (reject: numeric deviation)", op.name());
				println!(
					"    s{}' props: {}",
					step + 1,
					fmt_concrete_props(task, &state_packer, &cand_buffer, &prop_scope, 80)
				);
				println!(
					"    s{}' nums:  {}",
					step + 1,
					fmt_concrete_nums(&cand_numeric, &num_scope, partitions, 80)
				);
			}
		}

		let Some((_op_id, next_buffer, next_numeric)) = chosen else {
			println!("  step {step}: no applicable concrete operator found for wildcard options");
			break;
		};
		buffer = next_buffer;
		numeric_state = next_numeric;

		println!(
			"  s{} props: {}",
			step + 1,
			fmt_concrete_props(task, &state_packer, &buffer, &prop_scope, 200)
		);
		println!(
			"  s{} nums:  {}",
			step + 1,
			fmt_concrete_nums(&numeric_state, &num_scope, partitions, 200)
		);
	}
}

fn trace_variable_scope(
	task: &dyn AbstractNumericTask,
	plan: &WildcardPlanResult,
	shown_steps: usize,
) -> (Vec<usize>, Vec<usize>) {
	let ops = task.get_operators();
	let mut prop_vars: BTreeSet<usize> = BTreeSet::new();
	let mut num_vars: BTreeSet<usize> = BTreeSet::new();

	for choices in plan.wildcard_plan.iter().take(shown_steps) {
		for &op_id in choices.iter() {
			let Some(op) = ops.get(op_id) else {
				continue;
			};
			for pre in op.preconditions().iter() {
				prop_vars.insert(pre.var() as usize);
			}
			for eff in op.effects().iter() {
				prop_vars.insert(eff.var_id() as usize);
				for c in eff.conditions().iter() {
					prop_vars.insert(c.var() as usize);
				}
			}
			for neff in op.assignment_effects().iter() {
				num_vars.insert(neff.var_id() as usize);
				num_vars.insert(neff.affected_var_id() as usize);
				for c in neff.conditions().iter() {
					prop_vars.insert(c.var() as usize);
				}
			}
		}
	}

	(prop_vars.into_iter().collect(), num_vars.into_iter().collect())
}

fn fmt_concrete_props(
	task: &dyn AbstractNumericTask,
	packer: &IntDoublePacker,
	buffer: &[u64],
	var_ids: &[usize],
	max_items: usize,
) -> String {
	let mut out = String::new();
	let mut shown = 0usize;
	for &var_id in var_ids.iter() {
		if shown >= max_items {
			let _ = write!(&mut out, " ...");
			break;
		}
		let dom = task.variables().get(var_id).map(|v| v.domain_size()).unwrap_or(0);
		if dom <= 1 {
			continue;
		}
		if shown > 0 {
			out.push(' ');
		}
		let val = packer.get(buffer, var_id as i32) as i32;
		let _ = write!(&mut out, "v{var_id}={val}");
		shown += 1;
	}
	if out.is_empty() {
		"<empty>".to_string()
	} else {
		out
	}
}

fn fmt_concrete_nums(
	numeric_state: &[f64],
	var_ids: &[usize],
	partitions: &NumericPartitions,
	max_items: usize,
) -> String {
	let mut out = String::new();
	let mut shown = 0usize;
	for &num_id in var_ids.iter() {
		if shown >= max_items {
			let _ = write!(&mut out, " ...");
			break;
		}
		let Some(&v) = numeric_state.get(num_id) else {
			continue;
		};
		if shown > 0 {
			out.push(' ');
		}
		let mut part_s = String::new();
		if let Some(parts) = partitions.partitions(num_id) {
			if let Some(pid) = partition_for_value(parts, v) {
				let pid_u = usize::try_from(pid).unwrap_or(0);
				let iv_s = partitions
					.partition_interval(num_id, pid_u)
					.map(fmt_interval)
					.unwrap_or_else(|| "<missing-interval>".to_string());
				part_s = format!(" p{pid_u}:{iv_s}");
			}
		}
		let _ = write!(&mut out, "n{num_id}={}{}", fmt_f64_compact(v), part_s);
		shown += 1;
	}
	if out.is_empty() {
		"<empty>".to_string()
	} else {
		out
	}
}

fn debug_print_flaws(_task: &dyn AbstractNumericTask, flaws: &[Flaw]) {
	println!("[Flaws] count={}", flaws.len());
	let max = 200usize;
	let shown = flaws.len().min(max);
	for (i, flaw) in flaws.iter().take(shown).enumerate() {
		match flaw {
			Flaw::Propositional(pf) => {
				println!(
					"  {i}: PropFlaw fact=(var={}, val={}) deps={}",
					pf.fact.var(),
					pf.fact.value(),
					pf.dependent_numeric_flaws.len()
				);
				for (j, nf) in pf.dependent_numeric_flaws.iter().enumerate() {
					println!(
						"      - dep[{j}]: NumericFlaw var={} value={} include_in_lower={}",
						nf.numeric_var_id,
						nf.value,
						nf.include_in_lower
					);
				}
			}
			Flaw::Numeric(nf) => {
				println!(
					"  {i}: NumericFlaw var={} value={} include_in_lower={}",
					nf.numeric_var_id,
					nf.value,
					nf.include_in_lower
				);
			}
		}
	}
	if flaws.len() > max {
		println!("[Flaws] (truncated: showing {shown} of {})", flaws.len());
	}
}

fn debug_print_refinement_summary(
	before: Option<u128>,
	after: Option<u128>,
	domain_sizes: &[i32],
	numeric_domain_sizes: &[usize],
	refined: bool,
) {
	let before_s = before.map(|v| v.to_string()).unwrap_or_else(|| "<overflow>".to_string());
	let after_s = after.map(|v| v.to_string()).unwrap_or_else(|| "<overflow>".to_string());
	println!("[Refine] refined={refined} abstract_states: {before_s} -> {after_s}");

	let mut refined_props: Vec<(usize, i32)> = domain_sizes
		.iter()
		.enumerate()
		.filter_map(|(i, &s)| (s > 1).then_some((i, s)))
		.collect();
	refined_props.sort_by_key(|(i, _)| *i);
	let refined_nums: Vec<(usize, usize)> = numeric_domain_sizes
		.iter()
		.enumerate()
		.filter_map(|(i, &s)| (s > 1).then_some((i, s)))
		.collect();

	if !refined_props.is_empty() {
		let preview = 30usize;
		let mut line = String::new();
		let _ = write!(&mut line, "[Refine] propositional splits: {} vars", refined_props.len());
		for (i, s) in refined_props.iter().take(preview) {
			let _ = write!(&mut line, " v{i}=>{s}");
		}
		if refined_props.len() > preview {
			let _ = write!(&mut line, " ...");
		}
		println!("{line}");
	}
	if !refined_nums.is_empty() {
		let preview = 30usize;
		let mut line = String::new();
		let _ = write!(&mut line, "[Refine] numeric splits: {} vars", refined_nums.len());
		for (i, s) in refined_nums.iter().take(preview) {
			let _ = write!(&mut line, " n{i}=>{s}");
		}
		if refined_nums.len() > preview {
			let _ = write!(&mut line, " ...");
		}
		println!("{line}");
	}
}

fn fmt_sparse_i32(values: &[i32], max_items: usize) -> String {
	let mut out = String::new();
	let mut shown = 0usize;
	for (i, &v) in values.iter().enumerate() {
		if v == 0 {
			continue;
		}
		if shown >= max_items {
			let _ = write!(&mut out, " ...");
			break;
		}
		if shown > 0 {
			out.push(' ');
		}
		let _ = write!(&mut out, "{i}:{v}");
		shown += 1;
	}
	if out.is_empty() {
		"<all-zero>".to_string()
	} else {
		out
	}
}

fn fmt_delta_i32(prev: &[i32], cur: &[i32], max_items: usize) -> String {
	let mut out = String::new();
	let len = prev.len().min(cur.len());
	let mut shown = 0usize;
	for i in 0..len {
		let a = prev[i];
		let b = cur[i];
		if a == b {
			continue;
		}
		if shown >= max_items {
			let _ = write!(&mut out, " ...");
			break;
		}
		if shown > 0 {
			out.push(' ');
		}
		let _ = write!(&mut out, "{i}:{a}->{b}");
		shown += 1;
	}
	out
}

fn fmt_nontrivial_props(values: &[i32], domain_sizes: &[i32], max_items: usize) -> String {
	let mut out = String::new();
	let mut shown = 0usize;
	let len = values.len().min(domain_sizes.len());
	for var_id in 0..len {
		if domain_sizes[var_id] <= 1 {
			continue;
		}
		if shown >= max_items {
			let _ = write!(&mut out, " ...");
			break;
		}
		if shown > 0 {
			out.push(' ');
		}
		let _ = write!(&mut out, "v{var_id}:{}", values[var_id]);
		shown += 1;
	}
	if out.is_empty() {
		"<no-nontrivial-vars>".to_string()
	} else {
		out
	}
}

fn fmt_nontrivial_nums(
	values: &[i32],
	numeric_domain_sizes: &[usize],
	partitions: &NumericPartitions,
	max_items: usize,
) -> String {
	let mut out = String::new();
	let mut shown = 0usize;
	let len = values.len().min(numeric_domain_sizes.len());
	for num_id in 0..len {
		if numeric_domain_sizes[num_id] <= 1 {
			continue;
		}
		if shown >= max_items {
			let _ = write!(&mut out, " ...");
			break;
		}
		if shown > 0 {
			out.push(' ');
		}
		let part_i32 = values[num_id];
		let part = usize::try_from(part_i32).unwrap_or(0);
		let iv = partitions.partition_interval(num_id, part);
		let iv_s = iv.map(fmt_interval).unwrap_or_else(|| "<missing-interval>".to_string());
		let _ = write!(&mut out, "n{num_id}=p{part}:{iv_s}");
		shown += 1;
	}
	if out.is_empty() {
		"<no-nontrivial-vars>".to_string()
	} else {
		out
	}
}

fn fmt_delta_numeric_partitions(
	prev: &[i32],
	cur: &[i32],
	partitions: &NumericPartitions,
	max_items: usize,
) -> String {
	let mut out = String::new();
	let len = prev.len().min(cur.len());
	let mut shown = 0usize;
	for num_id in 0..len {
		let a = prev[num_id];
		let b = cur[num_id];
		if a == b {
			continue;
		}
		if shown >= max_items {
			let _ = write!(&mut out, " ...");
			break;
		}
		if shown > 0 {
			out.push(' ');
		}
		let a_u = usize::try_from(a).unwrap_or(0);
		let b_u = usize::try_from(b).unwrap_or(0);
		let a_iv = partitions.partition_interval(num_id, a_u);
		let b_iv = partitions.partition_interval(num_id, b_u);
		let a_s = a_iv.map(fmt_interval).unwrap_or_else(|| "<missing-interval>".to_string());
		let b_s = b_iv.map(fmt_interval).unwrap_or_else(|| "<missing-interval>".to_string());
		let _ = write!(&mut out, "n{num_id}:p{a_u}:{a_s}->p{b_u}:{b_s}");
		shown += 1;
	}
	out
}

fn fmt_interval(iv: Interval) -> String {
	let l = if iv.lower_closed { '[' } else { '(' };
	let r = if iv.upper_closed { ']' } else { ')' };
	let lo = fmt_f64_compact(iv.lower);
	let hi = fmt_f64_compact(iv.upper);
	format!("{l}{lo}, {hi}{r}")
}

fn fmt_f64_compact(v: f64) -> String {
	if v.is_nan() {
		return "NaN".to_string();
	}
	let mut s = format!("{v}");
	let is_scientific = s.contains('e') || s.contains('E');
	if !is_scientific {
		if let Some(dot) = s.find('.') {
			let (head, tail) = s.split_at(dot + 1);
			let trimmed_tail = tail.trim_end_matches('0');
			s = if trimmed_tail.is_empty() {
				head.trim_end_matches('.').to_string()
			} else {
				format!("{head}{trimmed_tail}")
			};
		}
	}
	s
}

fn identity_domain_mapping_and_sizes(task: &dyn AbstractNumericTask) -> Result<(DomainMapping, Vec<i32>)> {
	let num_vars_i32 = task.get_num_variables();
	ensure!(num_vars_i32 >= 0, "task.get_num_variables() must be non-negative");
	let num_vars = usize::try_from(num_vars_i32).context("num_vars does not fit usize")?;

	let mut domain_sizes: Vec<i32> = Vec::with_capacity(num_vars);
	let mut domain_mapping: DomainMapping = Vec::with_capacity(num_vars);

	for var in 0..num_vars {
		let var_i32 = i32::try_from(var).context("var index does not fit i32")?;
		let size = task
			.get_variable_domain_size(var_i32)
			.map_err(|e| anyhow::anyhow!(e.to_string()))
			.with_context(|| format!("get_variable_domain_size({var}) failed"))?;
		ensure!(size > 0, "non-positive domain size for var {var}: {size}");
		domain_sizes.push(size);

		let mut mapping: Vec<i32> = Vec::with_capacity(size as usize);
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

fn set_initial_prop_values(task: &dyn AbstractNumericTask, packer: &IntDoublePacker, buffer: &mut [u64]) {
	let init = task.get_initial_propositional_state_values();
	for (var_id, &val) in init.iter().enumerate() {
		packer.set(buffer, var_id as i32, val as u64);
	}
}

fn fact_is_true(fact: &Fact, packer: &IntDoublePacker, buffer: &[u64]) -> bool {
	let current = packer.get(buffer, fact.var() as i32) as i32;
	current == fact.value()
}

fn comparison_eval_code(v: Option<bool>) -> i32 {
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
	let mut lower_inputs: Vec<Interval> = concrete_values.iter().copied().map(Interval::singleton).collect();
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
	prop_var_id: i32,
	numeric_state: &[f64],
) -> Vec<NumericFlaw> {
	let Some(tree) = comparison_index.comparison_tree(prop_var_id) else {
		return vec![];
	};

	let mut out: Vec<NumericFlaw> = Vec::new();
	for dep_var_id in tree.regular_numeric_var_dependencies(task) {
		let Ok(dep_var_usize) = usize::try_from(dep_var_id) else {
			continue;
		};
		let Some(&concrete_value) = numeric_state.get(dep_var_usize) else {
			continue;
		};
		let include_in_lower = determine_include_in_lower(tree, dep_var_usize, concrete_value, numeric_state);

		if can_split_numeric_var(partitions, dep_var_usize, concrete_value, include_in_lower) {
			out.push(NumericFlaw {
				numeric_var_id: dep_var_usize,
				value: concrete_value,
				include_in_lower,
			});
		} else if can_split_numeric_var(partitions, dep_var_usize, concrete_value, !include_in_lower) {
			out.push(NumericFlaw {
				numeric_var_id: dep_var_usize,
				value: concrete_value,
				include_in_lower: !include_in_lower,
			});
		}
	}
	out
}

fn get_precondition_flaws(
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
			let prop_var_id = pre.var() as i32;
			let dependent_numeric_flaws = if comparison_index.is_comparison_axiom_variable(prop_var_id) {
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
	let mut seen: BTreeSet<Fact> = BTreeSet::new();
	let mut derived_goal_vars: BTreeSet<u32> = BTreeSet::new();
	for goal_id in 0..num_goals {
		let goal_fact = task.get_goal_fact(goal_id as i32);
		let goal_var = goal_fact.var();
		let goal_is_derived = task.axioms().iter().any(|ax| ax.var_id() == goal_var);
		if goal_is_derived {
			derived_goal_vars.insert(goal_var);
			continue;
		}
		if !fact_is_true(goal_fact, packer, buffer) && seen.insert(goal_fact.clone()) {
			let prop_var_id = goal_fact.var() as i32;
			let dependent_numeric_flaws = if comparison_index.is_comparison_axiom_variable(prop_var_id) {
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
				let prop_var_id = pre.var() as i32;
				let dependent_numeric_flaws = if comparison_index.is_comparison_axiom_variable(prop_var_id) {
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

fn apply_operator_to_state(
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
			packer.set(buffer, eff.var_id() as i32, eff.value() as u64);
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
		numeric_state[affected_var_id] = planners_sas::numeric::numeric_task::AssignmentOperation::apply(
			numeric_state[affected_var_id],
			eff.operation(),
			operand,
		);
	}
}

fn partition_for_value(partitions: &[super::comparison_expression::Interval], value: f64) -> Option<i32> {
	partitions
		.iter()
		.position(|iv| iv.contains(value))
		.and_then(|i| i32::try_from(i).ok())
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

fn get_numeric_deviation_flaws(
	op: &planners_sas::numeric::numeric_task::Operator,
	numeric_current_state: &[f64],
	numeric_successor_state: &[f64],
	abstract_numeric_successor_state: &[i32],
	partitions: &NumericPartitions,
) -> Vec<Flaw> {
	let mut flaws: Vec<Flaw> = Vec::new();

	let num_vars = numeric_successor_state.len().min(abstract_numeric_successor_state.len());
	for var_id in 0..num_vars {
		let operator_modified_var = op
			.assignment_effects()
			.iter()
			.any(|eff| eff.affected_var_id() as usize == var_id);
		if !operator_modified_var {
			continue;
		}

		let abstract_value = abstract_numeric_successor_state[var_id];
		let Some(parts) = partitions.partitions(var_id) else {
			continue;
		};
		let Some(correct_abstract_value) = partition_for_value(parts, numeric_successor_state[var_id]) else {
			continue;
		};
		if abstract_value == correct_abstract_value {
			continue;
		}

		let concrete_next_value = numeric_successor_state[var_id];
		let concrete_current_value = numeric_current_state.get(var_id).copied().unwrap_or(concrete_next_value);
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




