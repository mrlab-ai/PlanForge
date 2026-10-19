use super::*;

#[test]
fn random_walk_lengths_are_deterministic() {
    let mut left = SmallRng::seed_from_u64(7);
    let mut right = SmallRng::seed_from_u64(7);
    let left_lengths = (0..20)
        .map(|_| random_walk_length(110.0, 1.0, &mut left).unwrap())
        .collect::<Vec<_>>();
    let right_lengths = (0..20)
        .map(|_| random_walk_length(110.0, 1.0, &mut right).unwrap())
        .collect::<Vec<_>>();

    assert_eq!(left_lengths, right_lengths);
    assert!(left_lengths.iter().all(|&length| length <= 440));
}
