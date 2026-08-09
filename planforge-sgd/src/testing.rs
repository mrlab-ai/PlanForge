//! Shared test fixtures: a deterministic RNG and a random task generator.
//!
//! Both the exactness sweep and the tensor differential tests need the same
//! generator, and it must produce the awkward cases on purpose: several effects
//! on one variable within a single operator (agreeing and conflicting),
//! conditional effects, and effects carrying `precondition_value`.

#![cfg(test)]

use planforge_sas::axioms::PropositionalAxiom;
use planforge_sas::numeric_task::{
    Effect, ExplicitFact, ExplicitVariable, Metric, NumericRootTask, Operator,
};

/// Deterministic splitmix64, so failures are reproducible without pulling in an
/// RNG whose stream could change between versions.
pub struct Rng(u64);

impl Rng {
    /// Seed the generator.
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub fn below(&mut self, bound: usize) -> usize {
        (self.next_u64() % bound as u64) as usize
    }

    /// True with probability `numerator / 100`.
    pub fn percent(&mut self, numerator: u64) -> bool {
        self.next_u64() % 100 < numerator
    }
}

/// A random tiny task shaped like translator output: one derived
/// global-constraint variable first, then a few primary finite-domain variables.
pub fn random_task(rng: &mut Rng) -> NumericRootTask {
    let num_primary = 1 + rng.below(3);
    let domains: Vec<usize> = (0..num_primary).map(|_| 2 + rng.below(2)).collect();

    // var0 is derived, mirroring `new-axiom@0()`. Primary variables follow.
    let mut variables = vec![ExplicitVariable::new(
        2,
        String::from("var0"),
        vec![String::from("gc"), String::from("not-gc")],
        Some(0),
        1,
    )];
    for (index, &size) in domains.iter().enumerate() {
        variables.push(ExplicitVariable::new(
            size,
            format!("var{}", index + 1),
            (0..size).map(|value| format!("v{index}_{value}")).collect(),
            None,
            0,
        ));
    }

    // Task variable id of primary variable `index`.
    let task_var = |index: usize| index + 1;

    let num_operators = 1 + rng.below(4);
    let mut operators = Vec::with_capacity(num_operators);
    for op_index in 0..num_operators {
        let mut preconditions = Vec::new();
        for index in 0..num_primary {
            if rng.percent(40) {
                let value = rng.below(domains[index]);
                preconditions.push(ExplicitFact::new(task_var(index), value));
            }
        }

        let mut effects = Vec::new();
        for index in 0..num_primary {
            // Zero, one or two effects on the same variable. Two effects is the
            // case that breaks naive mass summation.
            let count = rng.below(3);
            for _ in 0..count {
                let mut conditions = Vec::new();
                for other in 0..num_primary {
                    if rng.percent(30) {
                        let value = rng.below(domains[other]);
                        conditions.push(ExplicitFact::new(task_var(other), value));
                    }
                }
                let required = if rng.percent(30) {
                    let value = rng.below(domains[index]);
                    // The SAS parser hoists an effect's `precondition_value`
                    // onto the operator's preconditions; the classical-fragment check verifies
                    // that invariant, so the generator must respect it.
                    preconditions.push(ExplicitFact::new(task_var(index), value));
                    Some(value)
                } else {
                    None
                };
                let value = rng.below(domains[index]);
                effects.push(Effect::new(conditions, task_var(index), required, value));
            }
        }

        operators.push(Operator::new(
            format!("op{op_index}"),
            preconditions,
            effects,
            Vec::new(),
            1,
        ));
    }

    let goal_var = rng.below(num_primary);
    let goals = vec![ExplicitFact::new(
        task_var(goal_var),
        rng.below(domains[goal_var]),
    )];

    let mut state = vec![1];
    for &size in &domains {
        state.push(rng.below(size));
    }

    NumericRootTask::new(
        4,
        Metric::new(true, None),
        variables,
        // No numeric variables at all: these tasks carry no costs and no
        // constants, which keeps `register_state` happy (it only accepts
        // regular and cost variables) and matches what the transcription uses.
        Vec::new(),
        goals,
        Vec::new(),
        state,
        Vec::new(),
        operators,
        vec![PropositionalAxiom::new(Vec::new(), 0, 1, 0)],
        Vec::new(),
        Vec::new(),
        ExplicitFact::new(0, 0),
    )
}
