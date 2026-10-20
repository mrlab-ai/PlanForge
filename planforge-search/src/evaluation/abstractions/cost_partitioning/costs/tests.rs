use planforge_sas::numeric_task::{
    ExplicitValueIndex, INF_VALUE, NEG_INF_VALUE, NumericValue, ONE_VALUE, ZERO_VALUE,
};

use super::*;
use crate::evaluation::abstractions::common::transition_system::{
    AbstractTransition, AbstractTransitionSystem,
};

#[test]
fn expired_scp_table_deadline_uses_shared_typed_error() {
    let error = ensure_scp_table_deadline(Some(Instant::now())).unwrap_err();

    assert!(crate::resource_limits::is_deadline_exceeded(&error));
}

#[test]
fn regional_overlay_handles_multiple_multidimensional_uncovered_pieces() {
    let region =
        |x: Interval, y: Interval| CartesianSet::with_all_props_constrained(Vec::new(), vec![x, y]);
    let first = region(
        Interval::closed(NumericValue::new(0.0), NumericValue::new(1.0)),
        Interval::closed(NumericValue::new(0.0), NumericValue::new(1.0)),
    );
    let second = region(
        Interval::closed(NumericValue::new(0.0), NumericValue::new(1.0)),
        Interval::closed(NumericValue::new(2.0), NumericValue::new(3.0)),
    );
    let mut usage = SourceSetUsage {
        cells: vec![
            SourceSetUsageCell {
                region: first.clone(),
                amount: 1.0,
            },
            SourceSetUsageCell {
                region: second.clone(),
                amount: 2.0,
            },
        ],
        index: RefCell::new(CellIndex::Stale),
    };

    usage.add(
        &region(
            Interval::closed(NumericValue::new(0.0), NumericValue::new(3.0)),
            Interval::closed(NumericValue::new(0.0), NumericValue::new(3.0)),
        ),
        3.0,
    );

    assert_eq!(usage.max_over(&first), 4.0);
    assert_eq!(usage.max_over(&second), 5.0);
    assert_eq!(
        usage.max_over(&region(
            Interval::closed(NumericValue::new(2.0), NumericValue::new(3.0)),
            Interval::closed(NumericValue::new(2.0), NumericValue::new(3.0)),
        )),
        3.0
    );
    assert!(source_set_usage_cells_are_disjoint(&usage.cells));
}

#[test]
fn full_cost_source_sets_use_overlap_cover_without_geometric_overlay() {
    let region = |lower, upper| {
        CartesianSet::with_all_props_constrained(Vec::new(), vec![Interval::closed(lower, upper)])
    };
    let source_set = |lower, upper| AbstractOperatorSourceSets {
        labels: vec![LabelSourceSet {
            concrete_op_id: OperatorIndex::new(0),
            source: Arc::new(region(lower, upper)),
        }],
    };
    let source_sets = vec![
        source_set(NumericValue::new(0.0), NumericValue::new(2.0)),
        source_set(NumericValue::new(1.0), NumericValue::new(3.0)),
    ];
    let mut residuals = RemainingCosts::from_operator_costs(&[1.0]);

    residuals
        .reduce_by_abstract_source_sets(0, &source_sets, &[1.0, 1.0])
        .unwrap();

    let residual = &residuals.operator_residuals[0];
    assert_eq!(residual.full_source_set_usage.cells.len(), 2);
    assert!(residual.source_set_usage.cells.is_empty());
    let overlapping = LabelSourceSet {
        concrete_op_id: OperatorIndex::new(0),
        source: Arc::new(region(NumericValue::new(1.5), NumericValue::new(1.5))),
    };
    let disjoint = LabelSourceSet {
        concrete_op_id: OperatorIndex::new(0),
        source: Arc::new(region(NumericValue::new(4.0), NumericValue::new(5.0))),
    };
    assert_eq!(residuals.cost_for_source_set(1, 0, &overlapping), 0.0);
    assert_eq!(residuals.cost_for_source_set(1, 0, &disjoint), 1.0);
}

fn two_state_transition_system() -> AbstractTransitionSystem {
    AbstractTransitionSystem {
        transitions: vec![AbstractTransition {
            transition_id: 0,
            abstract_op_id: 0,
            concrete_op_ids: vec![OperatorIndex::new(0)],
            source_hash: 0,
            target_hash: 1,
        }],
        duplicate_transition_attempts: 0,
        backward: vec![vec![], vec![0]],
        forward: vec![vec![0], vec![]],
        goal_facts: vec![],
        goal_state_hashes: vec![1],
        initial_state_hash: 0,
        hash_multipliers: vec![],
        numeric_domain_sizes: vec![],
        cartesian_sets: vec![cartesian_set(0).into(), cartesian_set(1).into()],
    }
}

#[test]
fn explicit_label_cost_partitioning_saturates_transition_graph() {
    let system = two_state_transition_system();
    let (distances, saturated) =
        build_explicit_label_cost_partitioning_table(&system, &[5.0], None, None).unwrap();

    assert_eq!(distances, vec![5.0, 0.0]);
    assert_eq!(saturated, vec![5.0]);
}

#[test]
fn explicit_regional_cost_partitioning_uses_source_sets() {
    let system = two_state_transition_system();
    let source_sets = vec![AbstractOperatorSourceSets {
        labels: vec![LabelSourceSet {
            concrete_op_id: OperatorIndex::new(0),
            source: cartesian_set(0).into(),
        }],
    }];
    let residual = RemainingCosts::from_operator_costs(&[5.0]);
    let (distances, saturated) = build_explicit_regional_cost_partitioning_table(
        &system,
        &source_sets,
        &residual,
        0,
        None,
        None,
    )
    .unwrap();

    assert_eq!(distances, vec![5.0, 0.0]);
    assert_eq!(
        saturated
            .entries()
            .iter()
            .map(|entry| entry.amount)
            .collect::<Vec<_>>(),
        vec![5.0]
    );
}

fn cartesian_set(value: usize) -> CartesianSet {
    CartesianSet::with_all_props_constrained(
        vec![vec![ExplicitValueIndex::from_usize(value)]],
        Vec::new(),
    )
}

fn numeric_cartesian_set(lower: NumericValue, upper: NumericValue) -> CartesianSet {
    CartesianSet::with_all_props_constrained(
        vec![vec![ExplicitValueIndex::new(0)]],
        vec![Interval::closed(lower, upper)],
    )
}

fn source_set(lower: NumericValue, upper: NumericValue) -> LabelSourceSet {
    source_set_for_op(OperatorIndex::new(0), lower, upper)
}

fn source_set_for_op(
    concrete_op_id: OperatorIndex,
    lower: NumericValue,
    upper: NumericValue,
) -> LabelSourceSet {
    LabelSourceSet {
        concrete_op_id,
        source: numeric_cartesian_set(lower, upper).into(),
    }
}

fn source_set_2d(
    concrete_op_id: OperatorIndex,
    first: Interval,
    second: Interval,
) -> LabelSourceSet {
    LabelSourceSet {
        concrete_op_id,
        source: CartesianSet::with_all_props_constrained(
            vec![vec![ExplicitValueIndex::new(0)]],
            vec![first, second],
        )
        .into(),
    }
}

fn abstract_regions_for_interval(
    lower: NumericValue,
    upper: NumericValue,
) -> AbstractOperatorSourceSets {
    AbstractOperatorSourceSets {
        labels: vec![source_set(lower, upper)],
    }
}

#[test]
fn source_set_reductions_apply_to_same_concrete_operator_only() {
    let mut residuals = RemainingCosts::from_operator_costs(&[10.0, 10.0]);
    let reduced = abstract_regions_for_interval(NumericValue::new(3.0), NumericValue::new(7.0));
    residuals
        .reduce_by_abstract_source_sets(0, std::slice::from_ref(&reduced), &[3.0])
        .unwrap();

    let query = source_set(NumericValue::new(5.0), NumericValue::new(8.0));
    assert_eq!(residuals.cost_for_source_set(1, 0, &query), 7.0);
    let other_op_query = source_set_for_op(
        OperatorIndex::new(1),
        NumericValue::new(5.0),
        NumericValue::new(8.0),
    );
    assert_eq!(residuals.cost_for_source_set(1, 0, &other_op_query), 10.0);
}

#[test]
fn source_set_reduction_allows_full_cost() {
    let mut residuals = RemainingCosts::from_operator_costs(&[1.0]);
    let reduced = abstract_regions_for_interval(NumericValue::new(3.0), NumericValue::new(7.0));
    residuals
        .reduce_by_abstract_source_sets(0, std::slice::from_ref(&reduced), &[1.0])
        .unwrap();

    assert_eq!(residuals.cost_for_source_set(1, 0, &reduced.labels[0]), 0.0);
}

#[test]
fn same_abstract_operator_alternative_source_sets_do_not_stack() {
    let mut residuals = RemainingCosts::from_operator_costs(&[1.0]);
    let reduced = AbstractOperatorSourceSets {
        labels: vec![
            source_set(NumericValue::new(0.0), NumericValue::new(10.0)),
            source_set(NumericValue::new(5.0), NumericValue::new(15.0)),
        ],
    };
    residuals
        .reduce_by_abstract_source_sets(0, &[reduced], &[0.4])
        .unwrap();

    assert_eq!(
        residuals.cost_for_source_set(
            1,
            0,
            &source_set(NumericValue::new(7.0), NumericValue::new(8.0))
        ),
        0.6
    );
}

#[test]
fn disjoint_source_set_sources_do_not_reduce_residual_cost() {
    let mut residuals = RemainingCosts::from_operator_costs(&[10.0]);
    residuals
        .reduce_by_abstract_source_sets(
            0,
            &[abstract_regions_for_interval(
                NumericValue::new(0.0),
                NumericValue::new(2.0),
            )],
            &[4.0],
        )
        .unwrap();

    assert_eq!(
        residuals.cost_for_source_set(
            1,
            0,
            &source_set(NumericValue::new(3.0), NumericValue::new(5.0))
        ),
        10.0
    );
}

#[test]
fn target_hull_overlap_is_ignored_for_abstract_source_sets() {
    let mut residuals = RemainingCosts::from_operator_costs(&[10.0]);
    residuals
        .reduce_by_abstract_source_sets(
            0,
            &[abstract_regions_for_interval(
                NumericValue::new(1.0),
                NumericValue::new(10.0),
            )],
            &[4.0],
        )
        .unwrap();

    assert_eq!(
        residuals.cost_for_source_set(
            1,
            0,
            &source_set(NumericValue::new(10.5), NumericValue::new(11.0))
        ),
        10.0
    );
}

#[test]
fn overlapping_source_set_sources_reduce_residual_cost() {
    let mut residuals = RemainingCosts::from_operator_costs(&[10.0]);
    residuals
        .reduce_by_abstract_source_sets(
            0,
            &[
                abstract_regions_for_interval(NumericValue::new(0.0), NumericValue::new(5.0)),
                abstract_regions_for_interval(NumericValue::new(4.0), NumericValue::new(10.0)),
            ],
            &[3.0, 4.0],
        )
        .unwrap();

    assert_eq!(
        residuals.cost_for_source_set(
            1,
            0,
            &source_set(NumericValue::new(4.5), NumericValue::new(4.75))
        ),
        6.0
    );
}

#[test]
fn label_cp_steals_shared_operator_cost() {
    let mut residuals = RemainingCosts::from_operator_costs(&[1.0]);
    residuals
        .reduce_by_abstract_source_sets(
            0,
            &[abstract_regions_for_interval(
                NumericValue::new(0.0),
                NumericValue::new(5.0),
            )],
            &[1.0],
        )
        .unwrap();

    // Label CP has only one scalar residual for `go_east`: once the first
    // abstraction saturates it, every later abstraction sees zero.
    assert_eq!(residuals.operator_costs_for_label_cp(), vec![0.0]);
}

#[test]
fn region_cp_preserves_residual_for_a_disjoint_abstraction() {
    let mut residuals = RemainingCosts::from_operator_costs(&[1.0]);
    residuals
        .reduce_by_abstract_source_sets(
            0,
            &[abstract_regions_for_interval(
                NumericValue::new(0.0),
                NumericValue::new(5.0),
            )],
            &[1.0],
        )
        .unwrap();

    // The disjoint abstraction starts after the first one's active
    // source region, so region CP preserves the unit residual there.
    let disjoint = source_set(NumericValue::new(5.0 + 1e-6), NumericValue::new(10.0));
    let region_residual = residuals.cost_for_source_set(1, 0, &disjoint);
    assert_eq!(region_residual, 1.0);
    assert!(region_residual > residuals.operator_costs_for_label_cp()[0]);
    assert!(region_residual <= 11.0);
}

#[test]
fn region_cp_overlapping_nested_targets_order_insensitive() {
    fn move_source_sets(start: usize, end: usize) -> Vec<AbstractOperatorSourceSets> {
        (start..end)
            .map(|i| AbstractOperatorSourceSets {
                labels: vec![LabelSourceSet {
                    concrete_op_id: OperatorIndex::new(0),
                    source: CartesianSet::with_all_props_constrained(
                        vec![vec![ExplicitValueIndex::new(0)]],
                        vec![Interval::new(
                            NumericValue::new(i as f64),
                            NumericValue::new((i + 1) as f64),
                            false,
                            true,
                        )],
                    )
                    .into(),
                }],
            })
            .collect()
    }

    fn save_source_set(save_op_id: OperatorIndex) -> AbstractOperatorSourceSets {
        AbstractOperatorSourceSets {
            labels: vec![source_set_for_op(
                save_op_id,
                NumericValue::new(0.0),
                NumericValue::new(15.0),
            )],
        }
    }

    fn contribution(
        residuals: &RemainingCosts,
        abstraction_id: usize,
        source_sets: &[AbstractOperatorSourceSets],
    ) -> f64 {
        source_sets
            .iter()
            .enumerate()
            .map(|(abstract_op_id, source_set)| {
                source_set
                    .labels
                    .iter()
                    .map(|label| {
                        residuals.cost_for_source_set(abstraction_id, abstract_op_id, label)
                    })
                    .fold(f64::INFINITY, f64::min)
            })
            .sum()
    }

    fn reduce(
        residuals: &mut RemainingCosts,
        abstraction_id: usize,
        source_sets: &[AbstractOperatorSourceSets],
    ) {
        residuals
            .reduce_by_abstract_source_sets(
                abstraction_id,
                source_sets,
                &vec![1.0; source_sets.len()],
            )
            .unwrap();
    }

    let mut alpha10 = move_source_sets(0, 10);
    alpha10.push(save_source_set(OperatorIndex::new(1)));
    let mut alpha15 = move_source_sets(0, 15);
    alpha15.push(save_source_set(OperatorIndex::new(2)));

    let label_cp_value = {
        let mut residuals = RemainingCosts::from_operator_costs(&[1.0, 1.0, 1.0]);
        reduce(&mut residuals, 0, &alpha10);
        11.0 + residuals.operator_costs_for_label_cp()[2]
    };
    assert_eq!(label_cp_value, 12.0);

    let alpha10_then_alpha15 = {
        let mut residuals = RemainingCosts::from_operator_costs(&[1.0, 1.0, 1.0]);
        let first = contribution(&residuals, 0, &alpha10);
        reduce(&mut residuals, 0, &alpha10);
        let second = contribution(&residuals, 1, &alpha15);
        first + second
    };
    let alpha15_then_alpha10 = {
        let mut residuals = RemainingCosts::from_operator_costs(&[1.0, 1.0, 1.0]);
        let first = contribution(&residuals, 0, &alpha15);
        reduce(&mut residuals, 0, &alpha15);
        let second = contribution(&residuals, 1, &alpha10);
        first + second
    };

    assert_eq!(alpha10_then_alpha15, 17.0);
    assert_eq!(alpha15_then_alpha10, 17.0);
    assert!(alpha10_then_alpha15 <= 17.0);
    assert!(alpha15_then_alpha10 <= 17.0);
    assert!(alpha10_then_alpha15 >= 16.0);
    assert!(alpha15_then_alpha10 >= 16.0);
    assert!(alpha10_then_alpha15 > label_cp_value);
    assert!(alpha15_then_alpha10 > label_cp_value);
}

#[test]
fn cross_dimension_residual_shared() {
    let mut residuals = RemainingCosts::from_operator_costs(&[1.0]);
    let x_abstraction = AbstractOperatorSourceSets {
        labels: vec![source_set_2d(
            OperatorIndex::new(0),
            Interval::closed(NumericValue::new(0.0), NumericValue::new(1.0)),
            Interval::unbounded(),
        )],
    };
    residuals
        .reduce_by_abstract_source_sets(0, &[x_abstraction], &[1.0])
        .unwrap();

    let y_abstraction = source_set_2d(
        OperatorIndex::new(0),
        Interval::unbounded(),
        Interval::closed(NumericValue::new(0.0), NumericValue::new(1.0)),
    );
    assert_eq!(residuals.cost_for_source_set(1, 0, &y_abstraction), 0.0);
}

#[test]
fn infinite_tail_reduction_preserves_disjoint_tail_cost() {
    let mut residuals = RemainingCosts::from_operator_costs(&[10.0]);
    let tail = abstract_regions_for_interval(NEG_INF_VALUE, ZERO_VALUE);
    residuals
        .reduce_by_abstract_source_sets(0, &[tail], &[4.0])
        .unwrap();

    assert_eq!(
        residuals.cost_for_source_set(1, 0, &source_set(ONE_VALUE, INF_VALUE),),
        10.0
    );
    assert_eq!(
        residuals.cost_for_source_set(1, 0, &source_set(NEG_INF_VALUE, INF_VALUE)),
        6.0
    );
}

#[test]
fn open_infinite_tail_does_not_consume_boundary() {
    let mut residuals = RemainingCosts::from_operator_costs(&[1.0]);
    let open_tail = AbstractOperatorSourceSets {
        labels: vec![LabelSourceSet {
            concrete_op_id: OperatorIndex::new(0),
            source: CartesianSet::with_all_props_constrained(
                vec![vec![ExplicitValueIndex::new(0)]],
                vec![Interval::new(NEG_INF_VALUE, ZERO_VALUE, false, false)],
            )
            .into(),
        }],
    };
    residuals
        .reduce_by_abstract_source_sets(0, &[open_tail], &[1.0])
        .unwrap();

    assert_eq!(
        residuals.cost_for_source_set(
            1,
            0,
            &source_set(NumericValue::new(0.0), NumericValue::new(0.0))
        ),
        1.0
    );
}

#[test]
fn multidimensional_disjoint_regions_preserve_full_cost() {
    let mut residuals = RemainingCosts::from_operator_costs(&[10.0]);
    let lower_y = AbstractOperatorSourceSets {
        labels: vec![source_set_2d(
            OperatorIndex::new(0),
            Interval::closed(ZERO_VALUE, NumericValue::new(10.0)),
            Interval::new(NEG_INF_VALUE, ZERO_VALUE, false, true),
        )],
    };
    residuals
        .reduce_by_abstract_source_sets(0, &[lower_y], &[4.0])
        .unwrap();

    let upper_y = source_set_2d(
        OperatorIndex::new(0),
        Interval::closed(ZERO_VALUE, NumericValue::new(10.0)),
        Interval::new(ZERO_VALUE, INF_VALUE, false, false),
    );
    assert_eq!(residuals.cost_for_source_set(1, 0, &upper_y), 10.0);
}

#[test]
fn perpendicular_tail_allocations_preserve_untouched_corner() {
    let mut residuals = RemainingCosts::from_operator_costs(&[10.0]);
    let left = AbstractOperatorSourceSets {
        labels: vec![source_set_2d(
            OperatorIndex::new(0),
            Interval::new(NEG_INF_VALUE, ZERO_VALUE, false, true),
            Interval::unbounded(),
        )],
    };
    let lower = AbstractOperatorSourceSets {
        labels: vec![source_set_2d(
            OperatorIndex::new(0),
            Interval::unbounded(),
            Interval::new(NEG_INF_VALUE, ZERO_VALUE, false, true),
        )],
    };
    residuals
        .reduce_by_abstract_source_sets(0, &[left], &[4.0])
        .unwrap();
    residuals
        .reduce_by_abstract_source_sets(1, &[lower], &[3.0])
        .unwrap();

    let upper_right = source_set_2d(
        OperatorIndex::new(0),
        Interval::new(ZERO_VALUE, INF_VALUE, false, false),
        Interval::new(ZERO_VALUE, INF_VALUE, false, false),
    );
    let lower_left = source_set_2d(
        OperatorIndex::new(0),
        Interval::new(NEG_INF_VALUE, ZERO_VALUE, false, true),
        Interval::new(NEG_INF_VALUE, ZERO_VALUE, false, true),
    );
    assert_eq!(residuals.cost_for_source_set(2, 0, &upper_right), 10.0);
    assert_eq!(residuals.cost_for_source_set(2, 0, &lower_left), 3.0);
}

#[test]
fn source_set_usage_index_matches_exact_overlap_across_blocks() {
    let mut usage = SourceSetUsage::default();
    for index in 0..96 {
        let region = numeric_cartesian_set(
            NumericValue::new(index as f64),
            NumericValue::new(index as f64 + 0.5),
        );
        usage.add(&region, (index % 7 + 1) as f64);
    }
    assert_eq!(usage.cells.len(), 96);

    let query = numeric_cartesian_set(NumericValue::new(30.25), NumericValue::new(66.25));
    let expected = usage
        .cells
        .iter()
        .filter(|cell| cell.region.overlaps(&query))
        .map(|cell| cell.amount)
        .fold(0.0, f64::max);
    assert_eq!(usage.max_over(&query), expected);
    assert!(usage.index.borrow().ready().is_some());
}
