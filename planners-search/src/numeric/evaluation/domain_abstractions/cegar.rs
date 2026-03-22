use std::time::{Duration, Instant};

use anyhow::{ensure, Context, Result};

use planners_sas::numeric::numeric_task::{AbstractNumericTask, Fact};

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

#[derive(Debug, Clone)]
pub struct CegarConfig {
	pub max_iterations: usize,
	pub max_time: Option<Duration>,
	pub use_wildcard_plans: bool,
	pub combine_labels: bool,
	pub enable_refinement: bool,
}

impl Default for CegarConfig {
	fn default() -> Self {
		Self {
			max_iterations: 10_000,
			max_time: None,
			use_wildcard_plans: true,
			combine_labels: true,
			enable_refinement: false,
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
		Ok(Self { config })
	}

	pub fn build_abstraction(&self, task: &dyn AbstractNumericTask) -> Result<CegarOutcome> {
		run_cegar(task, self.config.clone())
	}
}

pub fn run_cegar(task: &dyn AbstractNumericTask, config: CegarConfig) -> Result<CegarOutcome> {
	ensure!(config.max_iterations > 0, "max_iterations must be > 0");

	let start = Instant::now();

	// Initialization: numeric-fd starts from an initial domain mapping + initial numeric mapping.
	// We keep it minimal here: identity mapping for propositional vars and one unbounded partition
	// per numeric var.
	let (mut domain_mapping, mut domain_sizes) = identity_domain_mapping_and_sizes(task)
		.context("failed to build identity domain mapping")?;

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

		// TDODO: not implemented yet.
		#[allow(unreachable_code)]
		{
			// TODO: collect flaws from the (wildcard) abstract plan and concrete execution.
			let flaws: Vec<Flaw> = todo!("collect flaws (numeric-fd: get_flaws)");

			if flaws.is_empty() {
				break;
			}

			// TODO: refinement strategies (numeric-fd: fix_flaws + helpers)
			// This should mutate `domain_mapping` / `partitions` and update size vectors.
			let _refined: bool = todo!("refine abstraction from flaws");

			// TODO: update `numeric_domain_sizes` from `partitions` if numeric refinement happened.

			iteration += 1;
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


