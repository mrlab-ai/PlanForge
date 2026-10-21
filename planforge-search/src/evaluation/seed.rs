//! Random seeds of evaluation components, and how components derive seeds
//! from one another.
//!
//! A planner run is reproducible by default: two runs of the same
//! configuration on the same task must produce the same plan and the same
//! number of expansions. Seeding from the clock instead made order selection,
//! CEGAR refinement and plan extraction vary between runs, which shows up as
//! an expansion count that drifts for no visible reason. Randomised behaviour
//! is available by asking for a seed, not by omitting one.
//!
//! An unset `random_seed` does not mean the same seed everywhere: the random
//! split selection of CEGAR uses [`RANDOM_SPLIT_SEED`], CEGAR's seeded
//! tie-breaking between splits takes the first candidate, and every other
//! consumer uses [`DEFAULT_RANDOM_SEED`].

/// The seed of a randomised component without an explicit `random_seed`:
/// initial splits, collection restarts, SCP orders and sampling, and the
/// greedy PDB variable order.
pub const DEFAULT_RANDOM_SEED: u64 = 0;

/// The seed of CEGAR's random split selection without an explicit
/// `random_seed`, and the default `random_seed` of domain abstractions. Unlike
/// [`DEFAULT_RANDOM_SEED`], it is not used by the other consumers of an unset
/// seed.
pub(crate) const RANDOM_SPLIT_SEED: u64 = 2011;

/// One step of SplitMix64: a bijection that spreads every input bit over the
/// whole output.
pub(crate) fn mix_seed(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

/// The 64-bit FNV-1a hash of `text`, a seed that does not depend on the
/// process or the platform.
pub(crate) fn stable_text_seed(text: &str) -> u64 {
    text.bytes().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// The seed of variant `variant_id` of the member for goal `goal_id`.
pub(crate) fn derive_variant_seed(base_seed: u64, goal_id: usize, variant_id: usize) -> u64 {
    let goal = u64::try_from(goal_id).expect("goal id does not fit u64");
    let variant = u64::try_from(variant_id).expect("variant id does not fit u64");
    mix_seed(base_seed ^ goal.rotate_left(21) ^ variant.rotate_left(43))
}

#[cfg(test)]
mod tests {
    use super::{derive_variant_seed, stable_text_seed};

    #[test]
    fn variant_coordinates_produce_distinct_deterministic_seeds() {
        let seeds = [
            derive_variant_seed(7, 0, 0),
            derive_variant_seed(7, 0, 1),
            derive_variant_seed(7, 1, 0),
        ];
        assert_ne!(seeds[0], seeds[1]);
        assert_ne!(seeds[0], seeds[2]);
        assert_eq!(seeds[1], derive_variant_seed(7, 0, 1));
    }

    #[test]
    fn text_seeds_are_stable_and_distinguish_variable_names() {
        assert_eq!(stable_text_seed("x(b0)"), stable_text_seed("x(b0)"));
        assert_ne!(stable_text_seed("x(b0)"), stable_text_seed("x(b1)"));
    }
}
