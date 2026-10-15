use planforge_sas::numeric_task::{
    ExplicitValueIndex, INF_VALUE, NEG_INF_VALUE, NumericValue, ONE_VALUE, ZERO_VALUE,
};

use super::*;

#[test]
fn expired_scp_table_deadline_uses_shared_typed_error() {
    let error = ensure_scp_table_deadline(Some(Instant::now())).unwrap_err();

    assert!(crate::resource_limits::is_deadline_exceeded(&error));
}

#[test]
fn regional_overlay_handles_multiple_multidimensional_uncovered_pieces() {
    let region =
        |x: Interval, y: Interval| StateRegion::with_all_props_constrained(Vec::new(), vec![x, y]);
    let first = region(
        Interval::closed(NumericValue::new(0.0), NumericValue::new(1.0)),
        Interval::closed(NumericValue::new(0.0), NumericValue::new(1.0)),
    );
    let second = region(
        Interval::closed(NumericValue::new(0.0), NumericValue::new(1.0)),
        Interval::closed(NumericValue::new(2.0), NumericValue::new(3.0)),
    );
    let mut usage = RegionalUsage {
        cells: vec![
            RegionalUsageCell {
                region: first.clone(),
                amount: 1.0,
            },
            RegionalUsageCell {
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
    assert!(regional_usage_cells_are_disjoint(&usage.cells));
}

#[test]
fn full_cost_operator_regions_use_overlap_cover_without_geometric_overlay() {
    let region = |lower, upper| {
        StateRegion::with_all_props_constrained(Vec::new(), vec![Interval::closed(lower, upper)])
    };
    let operator_region = |lower, upper| AbstractOperatorRegions {
        labels: vec![OperatorRegion {
            concrete_op_id: OperatorIndex::new(0),
            source: Arc::new(region(lower, upper)),
        }],
    };
    let operator_regions = vec![
        operator_region(NumericValue::new(0.0), NumericValue::new(2.0)),
        operator_region(NumericValue::new(1.0), NumericValue::new(3.0)),
    ];
    let mut residuals = TransitionResidualCosts::from_operator_costs(&[1.0]);

    residuals
        .reduce_by_abstract_operator_regions(
            0,
            &operator_regions,
            &AbstractOperatorCostFunction {
                operator_costs: vec![1.0, 1.0],
            },
        )
        .unwrap();

    let residual = &residuals.operator_residuals[0];
    assert_eq!(residual.full_regional_usage.cells.len(), 2);
    assert!(residual.regional_usage.cells.is_empty());
    let overlapping = OperatorRegion {
        concrete_op_id: OperatorIndex::new(0),
        source: Arc::new(region(NumericValue::new(1.5), NumericValue::new(1.5))),
    };
    let disjoint = OperatorRegion {
        concrete_op_id: OperatorIndex::new(0),
        source: Arc::new(region(NumericValue::new(4.0), NumericValue::new(5.0))),
    };
    assert_eq!(residuals.cost_for_operator_region(1, 0, &overlapping), 0.0);
    assert_eq!(residuals.cost_for_operator_region(1, 0, &disjoint), 1.0);
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
        state_regions: vec![state_region(0).into(), state_region(1).into()],
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
fn explicit_regional_cost_partitioning_uses_operator_regions() {
    let system = two_state_transition_system();
    let operator_regions = vec![AbstractOperatorRegions {
        labels: vec![OperatorRegion {
            concrete_op_id: OperatorIndex::new(0),
            source: state_region(0).into(),
        }],
    }];
    let residual = TransitionResidualCosts::from_operator_costs(&[5.0]);
    let (distances, saturated) = build_explicit_regional_cost_partitioning_table(
        &system,
        &operator_regions,
        &residual,
        0,
        None,
        None,
    )
    .unwrap();

    assert_eq!(distances, vec![5.0, 0.0]);
    assert_eq!(saturated.operator_costs, vec![5.0]);
}

fn state_region(value: usize) -> StateRegion {
    StateRegion::with_all_props_constrained(
        vec![vec![ExplicitValueIndex::from_usize(value)]],
        Vec::new(),
    )
}

fn numeric_state_region(lower: NumericValue, upper: NumericValue) -> StateRegion {
    StateRegion::with_all_props_constrained(
        vec![vec![ExplicitValueIndex::new(0)]],
        vec![Interval::closed(lower, upper)],
    )
}

fn operator_region(lower: NumericValue, upper: NumericValue) -> OperatorRegion {
    operator_region_for_op(OperatorIndex::new(0), lower, upper)
}

fn operator_region_for_op(
    concrete_op_id: OperatorIndex,
    lower: NumericValue,
    upper: NumericValue,
) -> OperatorRegion {
    OperatorRegion {
        concrete_op_id,
        source: numeric_state_region(lower, upper).into(),
    }
}

fn operator_region_2d(
    concrete_op_id: OperatorIndex,
    first: Interval,
    second: Interval,
) -> OperatorRegion {
    OperatorRegion {
        concrete_op_id,
        source: StateRegion::with_all_props_constrained(
            vec![vec![ExplicitValueIndex::new(0)]],
            vec![first, second],
        )
        .into(),
    }
}

fn abstract_regions_for_interval(
    lower: NumericValue,
    upper: NumericValue,
) -> AbstractOperatorRegions {
    AbstractOperatorRegions {
        labels: vec![operator_region(lower, upper)],
    }
}

#[test]
fn operator_region_reductions_apply_to_same_concrete_operator_only() {
    let mut residuals = TransitionResidualCosts::from_operator_costs(&[10.0, 10.0]);
    let reduced = abstract_regions_for_interval(NumericValue::new(3.0), NumericValue::new(7.0));
    residuals
        .reduce_by_abstract_operator_regions(
            0,
            std::slice::from_ref(&reduced),
            &AbstractOperatorCostFunction {
                operator_costs: vec![3.0],
            },
        )
        .unwrap();

    let query = operator_region(NumericValue::new(5.0), NumericValue::new(8.0));
    assert_eq!(residuals.cost_for_operator_region(1, 0, &query), 7.0);
    let other_op_query = operator_region_for_op(
        OperatorIndex::new(1),
        NumericValue::new(5.0),
        NumericValue::new(8.0),
    );
    assert_eq!(
        residuals.cost_for_operator_region(1, 0, &other_op_query),
        10.0
    );
}

#[test]
fn operator_region_reduction_allows_full_cost() {
    let mut residuals = TransitionResidualCosts::from_operator_costs(&[1.0]);
    let reduced = abstract_regions_for_interval(NumericValue::new(3.0), NumericValue::new(7.0));
    residuals
        .reduce_by_abstract_operator_regions(
            0,
            std::slice::from_ref(&reduced),
            &AbstractOperatorCostFunction {
                operator_costs: vec![1.0],
            },
        )
        .unwrap();

    assert_eq!(
        residuals.cost_for_operator_region(1, 0, &reduced.labels[0]),
        0.0
    );
}

#[test]
fn same_abstract_operator_alternative_operator_regions_do_not_stack() {
    let mut residuals = TransitionResidualCosts::from_operator_costs(&[1.0]);
    let reduced = AbstractOperatorRegions {
        labels: vec![
            operator_region(NumericValue::new(0.0), NumericValue::new(10.0)),
            operator_region(NumericValue::new(5.0), NumericValue::new(15.0)),
        ],
    };
    residuals
        .reduce_by_abstract_operator_regions(
            0,
            &[reduced],
            &AbstractOperatorCostFunction {
                operator_costs: vec![0.4],
            },
        )
        .unwrap();

    assert_eq!(
        residuals.cost_for_operator_region(
            1,
            0,
            &operator_region(NumericValue::new(7.0), NumericValue::new(8.0))
        ),
        0.6
    );
}

#[test]
fn disjoint_operator_region_sources_do_not_reduce_residual_cost() {
    let mut residuals = TransitionResidualCosts::from_operator_costs(&[10.0]);
    residuals
        .reduce_by_abstract_operator_regions(
            0,
            &[abstract_regions_for_interval(
                NumericValue::new(0.0),
                NumericValue::new(2.0),
            )],
            &AbstractOperatorCostFunction {
                operator_costs: vec![4.0],
            },
        )
        .unwrap();

    assert_eq!(
        residuals.cost_for_operator_region(
            1,
            0,
            &operator_region(NumericValue::new(3.0), NumericValue::new(5.0))
        ),
        10.0
    );
}

#[test]
fn target_hull_overlap_is_ignored_for_abstract_operator_regions() {
    let mut residuals = TransitionResidualCosts::from_operator_costs(&[10.0]);
    residuals
        .reduce_by_abstract_operator_regions(
            0,
            &[abstract_regions_for_interval(
                NumericValue::new(1.0),
                NumericValue::new(10.0),
            )],
            &AbstractOperatorCostFunction {
                operator_costs: vec![4.0],
            },
        )
        .unwrap();

    assert_eq!(
        residuals.cost_for_operator_region(
            1,
            0,
            &operator_region(NumericValue::new(10.5), NumericValue::new(11.0))
        ),
        10.0
    );
}

#[test]
fn overlapping_operator_region_sources_reduce_residual_cost() {
    let mut residuals = TransitionResidualCosts::from_operator_costs(&[10.0]);
    residuals
        .reduce_by_abstract_operator_regions(
            0,
            &[
                abstract_regions_for_interval(NumericValue::new(0.0), NumericValue::new(5.0)),
                abstract_regions_for_interval(NumericValue::new(4.0), NumericValue::new(10.0)),
            ],
            &AbstractOperatorCostFunction {
                operator_costs: vec![3.0, 4.0],
            },
        )
        .unwrap();

    assert_eq!(
        residuals.cost_for_operator_region(
            1,
            0,
            &operator_region(NumericValue::new(4.5), NumericValue::new(4.75))
        ),
        6.0
    );
}

#[test]
fn label_cp_steals_shared_operator_cost() {
    let mut residuals = TransitionResidualCosts::from_operator_costs(&[1.0]);
    residuals
        .reduce_by_abstract_operator_regions(
            0,
            &[abstract_regions_for_interval(
                NumericValue::new(0.0),
                NumericValue::new(5.0),
            )],
            &AbstractOperatorCostFunction {
                operator_costs: vec![1.0],
            },
        )
        .unwrap();

    // Label CP has only one scalar residual for `go_east`: once the first
    // abstraction saturates it, every later abstraction sees zero.
    assert_eq!(residuals.operator_costs_for_label_cp(), vec![0.0]);
}

#[test]
fn region_cp_preserves_residual_for_complementary_abstraction() {
    let mut residuals = TransitionResidualCosts::from_operator_costs(&[1.0]);
    residuals
        .reduce_by_abstract_operator_regions(
            0,
            &[abstract_regions_for_interval(
                NumericValue::new(0.0),
                NumericValue::new(5.0),
            )],
            &AbstractOperatorCostFunction {
                operator_costs: vec![1.0],
            },
        )
        .unwrap();

    // The complementary abstraction starts after the first one's active
    // source region, so region CP preserves the unit residual there.
    let complementary = operator_region(NumericValue::new(5.0 + 1e-6), NumericValue::new(10.0));
    let region_residual = residuals.cost_for_operator_region(1, 0, &complementary);
    assert_eq!(region_residual, 1.0);
    assert!(region_residual > residuals.operator_costs_for_label_cp()[0]);
    assert!(region_residual <= 11.0);
}

#[test]
fn region_cp_overlapping_nested_targets_order_insensitive() {
    fn move_operator_regions(start: usize, end: usize) -> Vec<AbstractOperatorRegions> {
        (start..end)
            .map(|i| AbstractOperatorRegions {
                labels: vec![OperatorRegion {
                    concrete_op_id: OperatorIndex::new(0),
                    source: StateRegion::with_all_props_constrained(
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

    fn save_operator_region(save_op_id: OperatorIndex) -> AbstractOperatorRegions {
        AbstractOperatorRegions {
            labels: vec![operator_region_for_op(
                save_op_id,
                NumericValue::new(0.0),
                NumericValue::new(15.0),
            )],
        }
    }

    fn contribution(
        residuals: &TransitionResidualCosts,
        abstraction_id: usize,
        operator_regions: &[AbstractOperatorRegions],
    ) -> f64 {
        operator_regions
            .iter()
            .enumerate()
            .map(|(abstract_op_id, operator_region)| {
                operator_region
                    .labels
                    .iter()
                    .map(|label| {
                        residuals.cost_for_operator_region(abstraction_id, abstract_op_id, label)
                    })
                    .fold(f64::INFINITY, f64::min)
            })
            .sum()
    }

    fn reduce(
        residuals: &mut TransitionResidualCosts,
        abstraction_id: usize,
        operator_regions: &[AbstractOperatorRegions],
    ) {
        residuals
            .reduce_by_abstract_operator_regions(
                abstraction_id,
                operator_regions,
                &AbstractOperatorCostFunction {
                    operator_costs: vec![1.0; operator_regions.len()],
                },
            )
            .unwrap();
    }

    let mut alpha10 = move_operator_regions(0, 10);
    alpha10.push(save_operator_region(OperatorIndex::new(1)));
    let mut alpha15 = move_operator_regions(0, 15);
    alpha15.push(save_operator_region(OperatorIndex::new(2)));

    let label_cp_value = {
        let mut residuals = TransitionResidualCosts::from_operator_costs(&[1.0, 1.0, 1.0]);
        reduce(&mut residuals, 0, &alpha10);
        11.0 + residuals.operator_costs_for_label_cp()[2]
    };
    assert_eq!(label_cp_value, 12.0);

    let alpha10_then_alpha15 = {
        let mut residuals = TransitionResidualCosts::from_operator_costs(&[1.0, 1.0, 1.0]);
        let first = contribution(&residuals, 0, &alpha10);
        reduce(&mut residuals, 0, &alpha10);
        let second = contribution(&residuals, 1, &alpha15);
        first + second
    };
    let alpha15_then_alpha10 = {
        let mut residuals = TransitionResidualCosts::from_operator_costs(&[1.0, 1.0, 1.0]);
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
    let mut residuals = TransitionResidualCosts::from_operator_costs(&[1.0]);
    let x_abstraction = AbstractOperatorRegions {
        labels: vec![operator_region_2d(
            OperatorIndex::new(0),
            Interval::closed(NumericValue::new(0.0), NumericValue::new(1.0)),
            Interval::unbounded(),
        )],
    };
    residuals
        .reduce_by_abstract_operator_regions(
            0,
            &[x_abstraction],
            &AbstractOperatorCostFunction {
                operator_costs: vec![1.0],
            },
        )
        .unwrap();

    let y_abstraction = operator_region_2d(
        OperatorIndex::new(0),
        Interval::unbounded(),
        Interval::closed(NumericValue::new(0.0), NumericValue::new(1.0)),
    );
    assert_eq!(
        residuals.cost_for_operator_region(1, 0, &y_abstraction),
        0.0
    );
}

#[test]
fn infinite_tail_reduction_preserves_disjoint_tail_cost() {
    let mut residuals = TransitionResidualCosts::from_operator_costs(&[10.0]);
    let tail = abstract_regions_for_interval(NEG_INF_VALUE, ZERO_VALUE);
    residuals
        .reduce_by_abstract_operator_regions(
            0,
            &[tail],
            &AbstractOperatorCostFunction {
                operator_costs: vec![4.0],
            },
        )
        .unwrap();

    assert_eq!(
        residuals.cost_for_operator_region(1, 0, &operator_region(ONE_VALUE, INF_VALUE),),
        10.0
    );
    assert_eq!(
        residuals.cost_for_operator_region(1, 0, &operator_region(NEG_INF_VALUE, INF_VALUE)),
        6.0
    );
}

#[test]
fn open_infinite_tail_does_not_consume_boundary() {
    let mut residuals = TransitionResidualCosts::from_operator_costs(&[1.0]);
    let open_tail = AbstractOperatorRegions {
        labels: vec![OperatorRegion {
            concrete_op_id: OperatorIndex::new(0),
            source: StateRegion::with_all_props_constrained(
                vec![vec![ExplicitValueIndex::new(0)]],
                vec![Interval::new(NEG_INF_VALUE, ZERO_VALUE, false, false)],
            )
            .into(),
        }],
    };
    residuals
        .reduce_by_abstract_operator_regions(
            0,
            &[open_tail],
            &AbstractOperatorCostFunction {
                operator_costs: vec![1.0],
            },
        )
        .unwrap();

    assert_eq!(
        residuals.cost_for_operator_region(
            1,
            0,
            &operator_region(NumericValue::new(0.0), NumericValue::new(0.0))
        ),
        1.0
    );
}

#[test]
fn multidimensional_disjoint_regions_preserve_full_cost() {
    let mut residuals = TransitionResidualCosts::from_operator_costs(&[10.0]);
    let lower_y = AbstractOperatorRegions {
        labels: vec![operator_region_2d(
            OperatorIndex::new(0),
            Interval::closed(ZERO_VALUE, NumericValue::new(10.0)),
            Interval::new(NEG_INF_VALUE, ZERO_VALUE, false, true),
        )],
    };
    residuals
        .reduce_by_abstract_operator_regions(
            0,
            &[lower_y],
            &AbstractOperatorCostFunction {
                operator_costs: vec![4.0],
            },
        )
        .unwrap();

    let upper_y = operator_region_2d(
        OperatorIndex::new(0),
        Interval::closed(ZERO_VALUE, NumericValue::new(10.0)),
        Interval::new(ZERO_VALUE, INF_VALUE, false, false),
    );
    assert_eq!(residuals.cost_for_operator_region(1, 0, &upper_y), 10.0);
}

#[test]
fn perpendicular_tail_allocations_preserve_untouched_corner() {
    let mut residuals = TransitionResidualCosts::from_operator_costs(&[10.0]);
    let left = AbstractOperatorRegions {
        labels: vec![operator_region_2d(
            OperatorIndex::new(0),
            Interval::new(NEG_INF_VALUE, ZERO_VALUE, false, true),
            Interval::unbounded(),
        )],
    };
    let lower = AbstractOperatorRegions {
        labels: vec![operator_region_2d(
            OperatorIndex::new(0),
            Interval::unbounded(),
            Interval::new(NEG_INF_VALUE, ZERO_VALUE, false, true),
        )],
    };
    residuals
        .reduce_by_abstract_operator_regions(
            0,
            &[left],
            &AbstractOperatorCostFunction {
                operator_costs: vec![4.0],
            },
        )
        .unwrap();
    residuals
        .reduce_by_abstract_operator_regions(
            1,
            &[lower],
            &AbstractOperatorCostFunction {
                operator_costs: vec![3.0],
            },
        )
        .unwrap();

    let upper_right = operator_region_2d(
        OperatorIndex::new(0),
        Interval::new(ZERO_VALUE, INF_VALUE, false, false),
        Interval::new(ZERO_VALUE, INF_VALUE, false, false),
    );
    let lower_left = operator_region_2d(
        OperatorIndex::new(0),
        Interval::new(NEG_INF_VALUE, ZERO_VALUE, false, true),
        Interval::new(NEG_INF_VALUE, ZERO_VALUE, false, true),
    );
    assert_eq!(residuals.cost_for_operator_region(2, 0, &upper_right), 10.0);
    assert_eq!(residuals.cost_for_operator_region(2, 0, &lower_left), 3.0);
}

#[test]
fn regional_usage_index_matches_exact_overlap_across_blocks() {
    let mut usage = RegionalUsage::default();
    for index in 0..96 {
        let region = numeric_state_region(
            NumericValue::new(index as f64),
            NumericValue::new(index as f64 + 0.5),
        );
        usage.add(&region, (index % 7 + 1) as f64);
    }
    assert_eq!(usage.cells.len(), 96);

    let query = numeric_state_region(NumericValue::new(30.25), NumericValue::new(66.25));
    let expected = usage
        .cells
        .iter()
        .filter(|cell| cell.region.overlaps(&query))
        .map(|cell| cell.amount)
        .fold(0.0, f64::max);
    assert_eq!(usage.max_over(&query), expected);
    assert!(usage.index.borrow().ready().is_some());
}
