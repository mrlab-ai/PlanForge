//! Split-selection policies from Schindler, Speck, and Helmert (ICAPS 2026).
//!
//! The artifact constructs one Cartesian abstraction for the original task,
//! replays the first flawed abstract plan, and selects a split randomly or by
//! the number of values excluded from the desired child. The generator owns
//! replay and refinement; this module keeps the artifact-specific policy
//! explicit so native collection generation does not silently inherit it.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Icaps26SplitSelection {
    Random,
    MinUnwanted,
    MaxUnwanted,
}

/// The artifact uses `std::mt19937`; preserving its stream makes RANDOM a
/// reproducible artifact policy rather than a new Rust-specific policy.
pub(super) struct ArtifactMt19937 {
    state: [u32; 624],
    index: usize,
}

impl ArtifactMt19937 {
    pub(super) fn new(seed: u32) -> Self {
        let mut state = [0; 624];
        state[0] = seed;
        for index in 1..state.len() {
            let previous = state[index - 1];
            state[index] = 1_812_433_253_u32
                .wrapping_mul(previous ^ (previous >> 30))
                .wrapping_add(index as u32);
        }
        Self { state, index: 624 }
    }

    fn next_u32(&mut self) -> u32 {
        if self.index == self.state.len() {
            for index in 0..self.state.len() {
                let joined = (self.state[index] & 0x8000_0000)
                    | (self.state[(index + 1) % self.state.len()] & 0x7fff_ffff);
                let mut twisted = joined >> 1;
                if joined & 1 != 0 {
                    twisted ^= 0x9908_b0df;
                }
                self.state[index] = self.state[(index + 397) % self.state.len()] ^ twisted;
            }
            self.index = 0;
        }
        let mut value = self.state[self.index];
        self.index += 1;
        value ^= value >> 11;
        value ^= (value << 7) & 0x9d2c_5680;
        value ^= (value << 15) & 0xefc6_0000;
        value ^ (value >> 18)
    }

    pub(super) fn uniform_index(&mut self, bound: usize) -> usize {
        assert!(bound > 0, "cannot sample from an empty ICAPS split set");
        let range = u32::try_from(bound).expect("ICAPS split set exceeds the artifact RNG range");
        // This is libstdc++'s unbiased 32-bit uniform_int_distribution.
        let threshold = range.wrapping_neg() % range;
        loop {
            let product = u64::from(self.next_u32()) * u64::from(range);
            if product as u32 >= threshold {
                return (product >> 32) as usize;
            }
        }
    }
}
