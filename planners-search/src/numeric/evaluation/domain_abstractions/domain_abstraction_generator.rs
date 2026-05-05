use std::rc::Rc;

use anyhow::{Context, Result, ensure};

use planners_sas::numeric::numeric_task::{AbstractNumericTask, NumericRootTask};

use super::abstract_operator_generator::AbstractOperator;
use super::abstracted_task::{DomainAbstractionTaskProjection, maybe_build_linear_abstracted_task};
use super::cegar::{Cegar, CegarConfig};
use super::domain_abstraction_factory::{AbstractDistanceTable, DomainAbstractionFactory};
use super::transition_cost_partitioning::AbstractOperatorFootprint;

/// Fully built abstraction artifact that can be used to evaluate concrete states.
#[derive(Debug, Clone)]
pub struct DomainAbstraction {
    pub factory: DomainAbstractionFactory,
    pub distance_table: AbstractDistanceTable,
    pub hash_multipliers: Vec<usize>,
    pub combine_labels: bool,
    pub task_projection: Option<DomainAbstractionTaskProjection>,
    pub transformed_task: Option<Rc<NumericRootTask>>,
    pub relevant_operator_ids: Vec<usize>,
    pub abstract_operators: Vec<AbstractOperator>,
    pub abstract_operator_footprints: Vec<AbstractOperatorFootprint>,
    pub metadata: DomainAbstractionMetadata,
}

#[derive(Debug, Clone, Default)]
pub struct DomainAbstractionMetadata {
    pub collection_iteration: Option<usize>,
    pub portfolio_strategy: Option<String>,
    pub flaw_kind: Option<String>,
    pub full_goal_task: Option<bool>,
    pub initial_seed_splits: Vec<String>,
    pub max_abstraction_size: Option<usize>,
}

impl DomainAbstraction {
    pub fn task_for_factory<'task>(
        &'task self,
        fallback: &'task dyn AbstractNumericTask,
    ) -> &'task dyn AbstractNumericTask {
        self.transformed_task
            .as_deref()
            .map(|task| task as &dyn AbstractNumericTask)
            .unwrap_or(fallback)
    }
}

#[derive(Debug, Clone)]
pub struct PreparedDomainAbstractionTask {
    pub transformed_task: Option<Rc<NumericRootTask>>,
    pub task_projection: Option<DomainAbstractionTaskProjection>,
}

impl PreparedDomainAbstractionTask {
    pub fn task_for<'task>(
        &'task self,
        fallback: &'task dyn AbstractNumericTask,
    ) -> &'task dyn AbstractNumericTask {
        self.transformed_task
            .as_deref()
            .map(|task| task as &dyn AbstractNumericTask)
            .unwrap_or(fallback)
    }
}

/// Numeric-fd style generator that constructs a domain abstraction via CEGAR.
#[derive(Debug, Clone)]
pub struct DomainAbstractionGenerator {
    cegar: Cegar,
    config: CegarConfig,
}

impl DomainAbstractionGenerator {
    pub fn new(config: CegarConfig) -> Result<Self> {
        let cegar = Cegar::new(config.clone()).context("failed to construct CEGAR")?;
        Ok(Self { cegar, config })
    }

    pub fn config(&self) -> &CegarConfig {
        &self.config
    }

    /// Builds a domain abstraction and its abstract distance table.
    pub fn generate(&self, task: &dyn AbstractNumericTask) -> Result<DomainAbstraction> {
        let prepared = prepare_domain_abstraction_task(task, self.config.transform_linear_task)?;
        self.generate_prepared(task, &prepared)
    }

    pub fn generate_prepared(
        &self,
        fallback_task: &dyn AbstractNumericTask,
        prepared: &PreparedDomainAbstractionTask,
    ) -> Result<DomainAbstraction> {
        let transformed_task = prepared.task_for(fallback_task);
        let outcome = self
            .cegar
            .build_abstraction(transformed_task)
            .context("CEGAR failed to build abstraction")?;

        let factory = outcome.final_state.factory;
        let mut operator_generator =
            factory.make_operator_generator(transformed_task, self.config.combine_labels)?;
        let abstract_operators = operator_generator
            .build_abstract_operators(transformed_task)
            .context("failed to build abstract operators")?;
        let abstract_operator_footprints = factory
            .build_abstract_operator_footprints(transformed_task, &abstract_operators)
            .context("failed to build abstract-operator footprints")?;
        let distance_table = factory
            .build_distance_table_with_operators(
                transformed_task,
                &operator_generator,
                &abstract_operators,
                false,
            )
            .context("failed to build abstract distance table")?;

        let hash_multipliers =
            compute_hash_multipliers(factory.domain_sizes(), factory.numeric_domain_sizes())
                .context("failed to compute hash multipliers")?;
        let mut relevant_operator_ids: Vec<usize> = abstract_operators
            .iter()
            .flat_map(|operator| operator.concrete_op_ids.iter().copied())
            .collect();
        relevant_operator_ids.sort_unstable();
        relevant_operator_ids.dedup();

        Ok(DomainAbstraction {
            factory,
            distance_table,
            hash_multipliers,
            combine_labels: self.config.combine_labels,
            task_projection: prepared.task_projection.clone(),
            transformed_task: prepared.transformed_task.clone(),
            relevant_operator_ids,
            abstract_operators,
            abstract_operator_footprints,
            metadata: DomainAbstractionMetadata::default(),
        })
    }
}

pub fn prepare_domain_abstraction_task(
    task: &dyn AbstractNumericTask,
    transform_linear_task: bool,
) -> Result<PreparedDomainAbstractionTask> {
    let abstracted_task = maybe_build_linear_abstracted_task(task, transform_linear_task)
        .context("failed to build abstracted task for domain abstraction")?;
    let (transformed_task, task_projection) = match abstracted_task {
        Some(abstracted_task) => {
            let (transformed_task, projection) = abstracted_task.into_parts();
            (Some(Rc::new(transformed_task)), Some(projection))
        }
        None => (None, None),
    };
    Ok(PreparedDomainAbstractionTask {
        transformed_task,
        task_projection,
    })
}

/// Computes mixed-radix hash multipliers for propositional and numeric variables.
///
/// This mirrors the hashing scheme used by `AbstractOperatorGenerator`.
pub fn compute_hash_multipliers(
    domain_sizes: &[usize],
    numeric_domain_sizes: &[usize],
) -> Result<Vec<usize>> {
    let total = domain_sizes
        .len()
        .checked_add(numeric_domain_sizes.len())
        .context("variable count overflow")?;
    ensure!(total > 0, "cannot compute hash multipliers for 0 variables");

    let mut multipliers: Vec<usize> = vec![0; total];
    let mut mult: usize = 1;
    for idx in 0..total {
        multipliers[idx] = mult;

        let radix: usize = if idx < domain_sizes.len() {
            domain_sizes[idx]
        } else {
            let n = idx - domain_sizes.len();
            *numeric_domain_sizes
                .get(n)
                .context("numeric domain size out of range")?
        };

        mult = mult
            .checked_mul(radix)
            .context("hash multiplier overflow")?;
    }

    Ok(multipliers)
}
