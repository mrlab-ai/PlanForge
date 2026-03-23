#[cfg(test)]
mod tests;

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::time::{Duration, Instant};

use anyhow::{ensure, Context, Result};

use planners_sas::numeric::axioms::AxiomEvaluator;
use planners_sas::numeric::numeric_task::{AbstractNumericTask, Fact};
use planners_sas::numeric::utils::int_packer::IntDoublePacker;

use super::abstract_operator_generator::DomainMapping;
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
					let operator_flaws = get_precondition_flaws(op, &state_packer, &buffer);
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
				let operator_flaws = get_precondition_flaws(op, &state_packer, &buffer);
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

		let goal_flaws = get_goal_flaws(task, &state_packer, &buffer);
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
			DependentNumericRefinement::None,
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
				.compute_wildcard_plan(task, config.combine_labels)
				.with_context(|| format!("failed to compute wildcard plan (iteration {iteration})"))?
		} else {
			let _table = factory
				.build_abstract_distance_table(task, config.combine_labels)
				.with_context(|| {
					format!("failed to build abstract distance table (iteration {iteration})")
				})?;
			None
		};

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
		if flaws.is_empty() {
			break;
		}

		let refined = cegar.fix_flaws(
			task,
			&flaws,
			&mut domain_mapping,
			&mut domain_sizes,
			&mut partitions,
			&mut numeric_domain_sizes,
		);
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

fn get_precondition_flaws(
	op: &planners_sas::numeric::numeric_task::Operator,
	packer: &IntDoublePacker,
	buffer: &[u64],
) -> Vec<Flaw> {
	let mut out: Vec<Flaw> = Vec::new();
	for pre in op.preconditions().iter() {
		if !fact_is_true(pre, packer, buffer) {
			out.push(Flaw::Propositional(PropFlaw {
				fact: pre.clone(),
				dependent_numeric_flaws: vec![],
			}));
		}
	}
	out
}

fn get_goal_flaws(task: &dyn AbstractNumericTask, packer: &IntDoublePacker, buffer: &[u64]) -> Vec<Flaw> {
	let num_goals_i32 = task.get_num_goals();
	let num_goals = usize::try_from(num_goals_i32.max(0)).unwrap_or(0);
	let mut out: Vec<Flaw> = Vec::new();
	for goal_id in 0..num_goals {
		let goal_fact = task.get_goal_fact(goal_id as i32);
		if !fact_is_true(goal_fact, packer, buffer) {
			out.push(Flaw::Propositional(PropFlaw {
				fact: goal_fact.clone(),
				dependent_numeric_flaws: vec![],
			}));
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




