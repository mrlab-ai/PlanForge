use anyhow::{ensure, Context, Result};

use planners_sas::numeric::numeric_task::AbstractNumericTask;

use super::cegar::{Cegar, CegarConfig};
use super::domain_abstraction_factory::{AbstractDistanceTable, DomainAbstractionFactory};

/// Fully built abstraction artifact that can be used to evaluate concrete states.
#[derive(Debug, Clone)]
pub struct DomainAbstraction {
    pub factory: DomainAbstractionFactory,
    pub distance_table: AbstractDistanceTable,
    pub hash_multipliers: Vec<i32>,
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
        let outcome = self
            .cegar
            .build_abstraction(task)
            .context("CEGAR failed to build abstraction")?;

        let factory = outcome.last_step.factory;
        let distance_table = factory
            .build_abstract_distance_table(task, self.config.combine_labels, false)
            .context("failed to build abstract distance table")?;

        let hash_multipliers = compute_hash_multipliers(
            factory.domain_sizes(),
            factory.numeric_domain_sizes(),
        )
        .context("failed to compute hash multipliers")?;

        Ok(DomainAbstraction {
            factory,
            distance_table,
            hash_multipliers,
        })
    }
}

/// Computes mixed-radix hash multipliers for propositional and numeric variables.
///
/// This mirrors the hashing scheme used by `AbstractOperatorGenerator`.
pub fn compute_hash_multipliers(
    domain_sizes: &[i32],
    numeric_domain_sizes: &[usize],
) -> Result<Vec<i32>> {
    let total = domain_sizes
        .len()
        .checked_add(numeric_domain_sizes.len())
        .context("variable count overflow")?;
    ensure!(total > 0, "cannot compute hash multipliers for 0 variables");

    let mut multipliers: Vec<i32> = vec![0; total];
    let mut mult: i64 = 1;
    for idx in 0..total {
        multipliers[idx] = i32::try_from(mult).context("hash multiplier does not fit i32")?;

        let radix_i64: i64 = if idx < domain_sizes.len() {
            let s = domain_sizes[idx];
            ensure!(s > 0, "domain size for var {idx} must be > 0");
            i64::from(s)
        } else {
            let n = idx - domain_sizes.len();
            let s = *numeric_domain_sizes
                .get(n)
                .context("numeric domain size out of range")?;
            let s_i64 = i64::try_from(s).context("numeric domain size does not fit i64")?;
            ensure!(s_i64 > 0, "numeric domain size for var {n} must be > 0");
            s_i64
        };

        mult = mult
            .checked_mul(radix_i64)
            .context("hash multiplier overflow")?;
        ensure!(
            mult <= i64::from(i32::MAX),
            "abstract state space too large for i32 hashing ({mult})"
        );
    }

    Ok(multipliers)
}

