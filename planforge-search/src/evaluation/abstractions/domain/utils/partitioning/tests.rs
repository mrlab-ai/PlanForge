use planforge_sas::numeric_task::{INF_VALUE, NAN_VALUE, NEG_INF_VALUE};

use super::*;

fn nv(value: f64) -> NumericValue {
    NumericValue::new(value)
}

/// `count` adjacent partitions with integer boundaries `0, 1, ..., count - 2`,
/// unbounded at both ends, alternating which side owns each boundary.
fn layout(count: usize) -> Vec<Interval> {
    assert!(count >= 2);
    let boundary = |index: usize| nv(index as f64);
    // Boundary `b` belongs to the lower partition when `b` is even.
    let lower_owns = |index: usize| index.is_multiple_of(2);
    let mut partitions = vec![Interval::new(
        NEG_INF_VALUE,
        boundary(0),
        false,
        lower_owns(0),
    )];
    for index in 1..count - 1 {
        partitions.push(Interval::new(
            boundary(index - 1),
            boundary(index),
            !lower_owns(index - 1),
            lower_owns(index),
        ));
    }
    partitions.push(Interval::new(
        boundary(count - 2),
        INF_VALUE,
        !lower_owns(count - 2),
        false,
    ));
    partitions
}

#[test]
fn boundary_values_map_by_their_closed_side_for_every_partition_count() {
    for count in [2, 3, 5, 8, 9, 17, 64, 257] {
        let partitions = layout(count);
        for index in 0..count - 1 {
            let boundary = index as f64;
            let expected = if index.is_multiple_of(2) {
                index
            } else {
                index + 1
            };
            assert_eq!(
                partition_for_value(&partitions, nv(boundary)),
                Some(expected),
                "{count} partitions, boundary {boundary}"
            );
            // Just inside either neighbour.
            assert_eq!(
                partition_for_value(&partitions, nv(boundary - 0.5)),
                Some(index)
            );
            assert_eq!(
                partition_for_value(&partitions, nv(boundary + 0.5)),
                Some(index + 1)
            );
        }
    }
}

#[test]
fn a_boundary_value_maps_to_the_same_partition_in_small_and_large_layouts() {
    let small = layout(4);
    let large = layout(300);
    for boundary in [0.0, 1.0, 2.0] {
        assert_eq!(
            partition_for_value(&small, nv(boundary)),
            partition_for_value(&large, nv(boundary)),
            "boundary {boundary}"
        );
    }
}

#[test]
fn values_are_compared_on_the_canonical_lattice_without_tolerance() {
    let partitions = vec![
        Interval::new(NEG_INF_VALUE, nv(0.3), false, false),
        Interval::new(nv(0.3), INF_VALUE, true, false),
    ];
    // 0.1 + 0.2 is 0.30000000000000004: the same lattice point as 0.3.
    assert_eq!(partition_for_value(&partitions, nv(0.1 + 0.2)), Some(1));
    assert_eq!(partition_for_value(&partitions, nv(0.3)), Some(1));
    // One lattice step below the boundary is below it.
    let below = nv(0.3 - float_tolerance::ABS_EPSILON);
    assert_eq!(partition_for_value(&partitions, below), Some(0));
}

#[test]
fn values_outside_a_bounded_layout_and_nan_have_no_partition() {
    let partitions = vec![Interval::singleton(nv(2.0))];
    assert_eq!(partition_for_value(&partitions, nv(2.0)), Some(0));
    assert_eq!(partition_for_value(&partitions, nv(3.0)), None);
    assert_eq!(partition_for_value(&partitions, NAN_VALUE), None);
    assert_eq!(partition_for_value(&layout(9), NAN_VALUE), None);
}
