use anyhow::{Context, Result, ensure};

use planners_sas::numeric::numeric_task::AbstractNumericTask;

use super::abstracted_task::{
    AbstractedTask, DomainAbstractionTaskProjection, maybe_build_linear_abstracted_task,
};
use super::cegar::{Cegar, CegarConfig};
use super::domain_abstraction_factory::{AbstractDistanceTable, DomainAbstractionFactory};

/// Fully built abstraction artifact that can be used to evaluate concrete states.
#[derive(Debug, Clone)]
pub struct DomainAbstraction {
    pub factory: DomainAbstractionFactory,
    pub distance_table: AbstractDistanceTable,
    pub hash_multipliers: Vec<usize>,
    pub combine_labels: bool,
    pub task_projection: Option<DomainAbstractionTaskProjection>,
    pub relevant_operator_ids: Vec<usize>,
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
        let abstracted_task =
            maybe_build_linear_abstracted_task(task, self.config.transform_linear_task)
                .context("failed to build abstracted task for domain abstraction")?;
        let (build_task, task_projection) = match abstracted_task {
            Some(abstracted_task) => {
                let projection = abstracted_task.projection();
                (
                    Box::new(abstracted_task) as Box<dyn AbstractedTask>,
                    Some(projection),
                )
            }
            None => (
                Box::new(BorrowedAbstractedTask {
                    task,
                    projection: DomainAbstractionTaskProjection::identity(task),
                }) as Box<dyn AbstractedTask>,
                None,
            ),
        };
        let transformed_task = build_task.task();
        let outcome = self
            .cegar
            .build_abstraction(transformed_task)
            .context("CEGAR failed to build abstraction")?;

        let factory = outcome.final_state.factory;
        let distance_table = factory
            .build_abstract_distance_table(transformed_task, self.config.combine_labels, false)
            .context("failed to build abstract distance table")?;

        let hash_multipliers =
            compute_hash_multipliers(factory.domain_sizes(), factory.numeric_domain_sizes())
                .context("failed to compute hash multipliers")?;
        let relevant_operator_ids = {
            let mut operator_generator =
                factory.make_operator_generator(transformed_task, self.config.combine_labels)?;
            let mut ids: Vec<usize> = operator_generator
                .build_abstract_operators(transformed_task)?
                .into_iter()
                .flat_map(|operator| operator.concrete_op_ids.into_iter())
                .collect();
            ids.sort_unstable();
            ids.dedup();
            ids
        };

        Ok(DomainAbstraction {
            factory,
            distance_table,
            hash_multipliers,
            combine_labels: self.config.combine_labels,
            task_projection,
            relevant_operator_ids,
        })
    }
}

struct BorrowedAbstractedTask<'task> {
    task: &'task dyn AbstractNumericTask,
    projection: DomainAbstractionTaskProjection,
}

impl AbstractedTask for BorrowedAbstractedTask<'_> {
    fn task(&self) -> &dyn AbstractNumericTask {
        self.task
    }

    fn project_state_values(
        &self,
        propositional_values: &[usize],
        numeric_values: &[f64],
    ) -> Result<super::abstracted_task::ProjectedState> {
        self.projection
            .project_state_values(propositional_values, numeric_values)
    }
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
