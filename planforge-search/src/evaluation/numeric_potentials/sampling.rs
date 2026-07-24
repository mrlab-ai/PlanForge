use planforge_sas::numeric_task::{AbstractNumericTask, TaskRef};
use planforge_sas::state_registry::{ConcreteState, StateRegistry};

use crate::successor_generator::SuccessorTree;

pub(crate) struct RandomWalkSampler {
    successor_generator: SuccessorTree,
    rng: CppMt19937,
    propositional: Vec<usize>,
    applicable: Vec<u32>,
    successor_numeric: Vec<f64>,
    successor_cost: Vec<f64>,
}

impl RandomWalkSampler {
    pub(crate) fn new(task: &dyn AbstractNumericTask, seed: u64) -> Self {
        Self {
            successor_generator: SuccessorTree::new(task),
            rng: CppMt19937::new(seed as u32),
            propositional: Vec::new(),
            applicable: Vec::new(),
            successor_numeric: Vec::new(),
            successor_cost: Vec::new(),
        }
    }

    pub(crate) fn sample(
        &mut self,
        task: &TaskRef<'_>,
        registry: &mut StateRegistry<'_>,
        count: usize,
        initial_h: f64,
        average_operator_cost: f64,
    ) -> Result<Vec<ConcreteState>, String> {
        let initial = registry.get_initial_state();
        let trials = if initial_h == 0.0 {
            10
        } else {
            if average_operator_cost == 0.0 {
                return Err(
                    "numeric_potential random walks require nonzero average operator cost when h(s0) is nonzero"
                        .to_string(),
                );
            }
            4 * ((initial_h / average_operator_cost) + 0.5) as usize
        };
        let mut samples = Vec::with_capacity(count);
        for _ in 0..count {
            let length = (0..trials).filter(|_| self.rng.real() < 0.5).count();
            let mut state = initial.clone();
            for _ in 0..length {
                state.fill_state(registry, &mut self.propositional);
                self.applicable.clear();
                self.successor_generator
                    .get_applicable_operators(&self.propositional, &mut self.applicable);
                if self.applicable.is_empty() {
                    break;
                }
                let operator_id = self.applicable[self.rng.index(self.applicable.len())];
                let operator = &task.get_operators()[operator_id as usize];
                state = registry
                    .get_successor_state_with_buffers(
                        &state,
                        operator,
                        &mut self.successor_numeric,
                        &mut self.successor_cost,
                    )
                    .map_err(|error| {
                        format!(
                            "numeric_potential random walk failed applying operator {} (`{}`): {error:?}",
                            operator_id,
                            operator.name()
                        )
                    })?;
            }
            samples.push(state);
        }
        Ok(samples)
    }

    pub(crate) fn choose_index(&mut self, bound: usize) -> usize {
        self.rng.index(bound)
    }
}

/// Bit-for-bit implementation of `std::mt19937` plus the libstdc++
/// `uniform_real_distribution<double>` and integer downscaling used by the
/// C++ planner. Keeping this local stream makes sample traces reproducible
/// across the two implementations.
struct CppMt19937 {
    state: [u32; 624],
    index: usize,
}

impl CppMt19937 {
    fn new(seed: u32) -> Self {
        let mut state = [0; 624];
        state[0] = seed;
        for i in 1..624 {
            state[i] = 1_812_433_253_u32
                .wrapping_mul(state[i - 1] ^ (state[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        Self { state, index: 624 }
    }

    fn next_u32(&mut self) -> u32 {
        if self.index == 624 {
            self.twist();
        }
        let mut value = self.state[self.index];
        self.index += 1;
        value ^= value >> 11;
        value ^= (value << 7) & 0x9d2c_5680;
        value ^= (value << 15) & 0xefc6_0000;
        value ^= value >> 18;
        value
    }

    fn twist(&mut self) {
        for i in 0..624 {
            let combined =
                (self.state[i] & 0x8000_0000) | (self.state[(i + 1) % 624] & 0x7fff_ffff);
            self.state[i] = self.state[(i + 397) % 624]
                ^ (combined >> 1)
                ^ if combined & 1 == 0 { 0 } else { 0x9908_b0df };
        }
        self.index = 0;
    }

    fn real(&mut self) -> f64 {
        const RANGE: f64 = 4_294_967_296.0;
        (self.next_u32() as f64 + self.next_u32() as f64 * RANGE) / (RANGE * RANGE)
    }

    fn index(&mut self, bound: usize) -> usize {
        assert!(bound > 0 && bound <= u32::MAX as usize);
        let scaling = u32::MAX as u64 / bound as u64;
        let past = scaling * bound as u64;
        loop {
            let value = self.next_u32() as u64;
            if value < past {
                return (value / scaling) as usize;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::CppMt19937;

    #[test]
    fn matches_libstdcpp_distributions() {
        let mut rng = CppMt19937::new(2011);
        let expected = [
            0.082_139_092_946_262_4,
            0.838_044_538_055_328_3,
            0.502_346_321_560_999_1,
            0.504_049_694_125_299,
            0.820_346_448_944_931_6,
        ];
        for expected in expected {
            assert_eq!(rng.real(), expected);
        }
        assert_eq!(
            (0..10).map(|_| rng.index(7)).collect::<Vec<_>>(),
            [5, 4, 3, 2, 2, 6, 2, 3, 1, 1]
        );
    }
}
