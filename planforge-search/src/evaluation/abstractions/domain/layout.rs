//! The abstract state space of a domain abstraction.

use anyhow::{Context, Result, ensure};
use planforge_sas::numeric_task::VariableIndex;

use super::abstraction_numeric_var;

/// The abstract states of a domain abstraction as mixed-radix numbers: one
/// digit per variable, the task's propositional variables first and its
/// numeric variables after them, the first digit least significant. A
/// variable's radix is its number of abstract values, and a state's id is the
/// number its digits spell, so the ids are exactly `0..num_states`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StateLayout {
    num_props: usize,
    radices: Vec<usize>,
    multipliers: Vec<usize>,
    num_states: usize,
}

impl StateLayout {
    /// The layout of an abstraction with `domain_sizes` abstract values per
    /// propositional variable and `numeric_domain_sizes` partitions per
    /// numeric variable. Fails if the state ids do not fit `isize`, which the
    /// signed offsets between them need.
    pub fn new(domain_sizes: &[usize], numeric_domain_sizes: &[usize]) -> Result<Self> {
        let radices: Vec<usize> = domain_sizes
            .iter()
            .chain(numeric_domain_sizes)
            .copied()
            .collect();
        let mut multipliers = Vec::with_capacity(radices.len());
        let mut num_states: usize = 1;
        for (var, &radix) in radices.iter().enumerate() {
            ensure!(radix > 0, "abstract variable {var} has no abstract value");
            multipliers.push(num_states);
            num_states = num_states
                .checked_mul(radix)
                .filter(|&states| isize::try_from(states).is_ok())
                .context("the abstract state space does not fit isize")?;
        }
        Ok(Self {
            num_props: domain_sizes.len(),
            radices,
            multipliers,
            num_states,
        })
    }

    pub fn num_states(&self) -> usize {
        self.num_states
    }

    /// The number of abstract values of every propositional variable.
    pub fn domain_sizes(&self) -> &[usize] {
        &self.radices[..self.num_props]
    }

    /// The number of partitions of every numeric variable.
    pub fn numeric_domain_sizes(&self) -> &[usize] {
        &self.radices[self.num_props..]
    }

    /// The number of abstract values of abstraction variable `var`.
    pub fn radix(&self, var: usize) -> usize {
        self.radices[var]
    }

    /// Per abstraction variable, what one unit of its value adds to a state
    /// id.
    pub fn multipliers(&self) -> &[usize] {
        &self.multipliers
    }

    /// The abstract value of abstraction variable `var` in `state`.
    pub fn value(&self, state: usize, var: usize) -> usize {
        debug_assert!(
            state < self.num_states,
            "abstract state {state} out of range"
        );
        (state / self.multipliers[var]) % self.radices[var]
    }

    /// The partition of numeric variable `numeric_var_id` in `state`.
    pub fn partition(&self, state: usize, numeric_var_id: VariableIndex) -> usize {
        self.value(
            state,
            abstraction_numeric_var(self.num_props, numeric_var_id).index(),
        )
    }

    /// The number of states after splitting one abstract value of `var` in
    /// two, exactly.
    pub fn states_after_split(&self, var: usize) -> u128 {
        let radix = self.radices[var] as u128;
        self.num_states as u128 / radix * (radix + 1)
    }
}
