use super::*;

#[derive(Debug, Clone, Copy, Default)]
pub(super) struct GoalCoverOrderVariant {
    pub(super) anchor_goal_offset: usize,
    pub(super) anchor_offset: usize,
    pub(super) second_anchor_round: usize,
    pub(super) representative_round: usize,
    pub(super) non_goal_prefix: bool,
    pub(super) compact: bool,
}

impl<'task> SaturatedCostPartitioningHeuristic<'task> {
    pub(super) fn goal_cover_order(
        &self,
        base_order: &[usize],
        components: &[AbstractionComponent<'_>],
        abstraction_h: &[f64],
        require_pure_goal_focused_collection: bool,
    ) -> Option<Vec<usize>> {
        goal_cover_order(
            base_order,
            components,
            abstraction_h,
            require_pure_goal_focused_collection,
            GoalCoverOrderVariant::default(),
        )
    }

    pub(super) fn compact_goal_cover_orders(
        &self,
        base_order: &[usize],
        components: &[AbstractionComponent<'_>],
        abstraction_h: &[f64],
    ) -> Vec<(usize, Vec<usize>)> {
        let mut variants_by_goal = HashMap::<usize, usize>::new();
        for member in components
            .iter()
            .filter_map(AbstractionComponent::goal_focused_member)
        {
            *variants_by_goal
                .entry(member.membership.goal_id)
                .or_default() += 1;
        }
        let progress_states = components
            .iter()
            .filter_map(AbstractionComponent::goal_focused_member)
            .any(|member| member.membership.from_progress_state);
        let goal_count = variants_by_goal
            .values()
            .filter(|&&variant_count| progress_states || variant_count >= 2)
            .count();
        if goal_count == 0 {
            return Vec::new();
        }
        let max_variants = variants_by_goal
            .values()
            .copied()
            .max()
            .expect("a goal with variants has a variant count");

        // Rotate the anchor goal before varying its construction variant, so
        // that every goal is covered, including states where that goal is the
        // last one remaining.
        let variants = compact_goal_cover_variants(
            goal_count,
            max_variants,
            progress_states.then_some(self.config.goal_cover_orders_per_goal),
        );

        let mut seen = HashSet::new();
        variants
            .into_iter()
            .filter_map(|variant| {
                let order =
                    goal_cover_order(base_order, components, abstraction_h, false, variant)?;
                let first_component = *order
                    .first()
                    .expect("compact goal-cover order must not be empty");
                let goal_id = components[first_component]
                    .goal_focused_member()
                    .expect("compact goal-cover order must start with its anchor goal")
                    .membership
                    .goal_id;
                seen.insert(order.clone()).then_some((goal_id, order))
            })
            .collect()
    }

    pub(super) fn goal_cover_goal_of_order(
        &self,
        order: &[usize],
        components: &[AbstractionComponent<'_>],
    ) -> Option<usize> {
        if order.len() != components.len()
            || !order
                .iter()
                .all(|&id| components[id].goal_focused_member().is_some())
        {
            return None;
        }
        let first = order.first().expect("an SCP collection has a component");
        let first = components[*first]
            .goal_focused_member()
            .expect("every component of the order is a goal-focused member");
        first
            .membership
            .from_progress_state
            .then_some(first.membership.goal_id)
    }

    pub(super) fn prefixed_goal_cover_order(
        &self,
        base_order: &[usize],
        components: &[AbstractionComponent<'_>],
        abstraction_h: &[f64],
    ) -> Option<Vec<usize>> {
        goal_cover_order(
            base_order,
            components,
            abstraction_h,
            false,
            GoalCoverOrderVariant {
                non_goal_prefix: true,
                ..Default::default()
            },
        )
    }
}

/// Every anchor goal with every pair of anchor and representative variants;
/// with `specialists_per_goal`, also at least that many anchor variants per
/// goal.
pub(super) fn compact_goal_cover_variants(
    goal_count: usize,
    variants_per_goal: usize,
    specialists_per_goal: Option<usize>,
) -> Vec<GoalCoverOrderVariant> {
    assert!(goal_count > 0);
    assert!(variants_per_goal > 0);
    let pairwise_variant_count = goal_count
        .checked_mul(variants_per_goal)
        .and_then(|count| count.checked_mul(variants_per_goal))
        .expect("the goal-cover variant count fits usize");
    let specialist_coverage_count = specialists_per_goal.map_or(0, |specialists| {
        goal_count * variants_per_goal.min(specialists)
    });
    let variant_count = pairwise_variant_count.max(specialist_coverage_count);
    (0..variant_count)
        .map(|variant_index| {
            let anchor_goal_offset = variant_index % goal_count;
            let anchor_round = variant_index / goal_count;
            let anchor_offset = anchor_round % variants_per_goal;
            let representative_round = (anchor_round / variants_per_goal) % variants_per_goal;
            GoalCoverOrderVariant {
                anchor_goal_offset,
                anchor_offset,
                second_anchor_round: anchor_offset.wrapping_add(representative_round),
                representative_round,
                compact: true,
                ..Default::default()
            }
        })
        .collect()
}

pub(super) fn goal_cover_order(
    base_order: &[usize],
    components: &[AbstractionComponent<'_>],
    abstraction_h: &[f64],
    require_pure_goal_focused_collection: bool,
    variant: GoalCoverOrderVariant,
) -> Option<Vec<usize>> {
    let goal_member = |component_id: usize| components[component_id].goal_focused_member();
    let is_goal_member = |component_id: usize| goal_member(component_id).is_some();
    let pure_goal_focused_collection = base_order.iter().copied().all(is_goal_member);
    if (require_pure_goal_focused_collection || variant.compact) && !pure_goal_focused_collection {
        return None;
    }

    let mut by_goal: HashMap<usize, Vec<usize>> = HashMap::new();
    for &component_id in base_order {
        let Some(member) = goal_member(component_id) else {
            continue;
        };
        by_goal
            .entry(member.membership.goal_id)
            .or_default()
            .push(component_id);
    }
    let progress_states = components
        .iter()
        .filter_map(AbstractionComponent::goal_focused_member)
        .any(|member| member.membership.from_progress_state);
    if by_goal.is_empty()
        || (!progress_states && by_goal.values().all(|components| components.len() < 2))
    {
        return None;
    }

    let current_h = |component_id: usize| abstraction_h[component_id];
    let member = |component_id: usize| {
        goal_member(component_id)
            .expect("goal-cover order component must reference a goal-focused collection member")
    };
    let goal_max_h = |components: &[usize]| {
        components
            .iter()
            .copied()
            .map(current_h)
            .fold(0.0, f64::max)
    };
    let mut sorted_goals = by_goal
        .iter()
        .filter(|(_, components)| progress_states || components.len() >= 2)
        .collect::<Vec<_>>();
    sorted_goals.sort_by(|(left_goal, left), (right_goal, right)| {
        goal_max_h(right)
            .total_cmp(&goal_max_h(left))
            .then_with(|| left_goal.cmp(right_goal))
    });
    let (&anchor_goal, anchor_components) =
        sorted_goals[variant.anchor_goal_offset % sorted_goals.len()];

    let compare_anchor = |&left: &usize, &right: &usize| {
        let left_member = member(left);
        let right_member = member(right);
        current_h(right)
            .total_cmp(&current_h(left))
            .then_with(|| right_member.tie_rank.cmp(&left_member.tie_rank))
            .then_with(|| {
                (right_member.split_kind == SplitKind::Wanted)
                    .cmp(&(left_member.split_kind == SplitKind::Wanted))
            })
            .then_with(|| left.cmp(&right))
    };
    let mut sorted_anchor_components = anchor_components.clone();
    sorted_anchor_components.sort_by(compare_anchor);
    let first_anchor =
        sorted_anchor_components[variant.anchor_offset % sorted_anchor_components.len()];
    let first_member = member(first_anchor);
    let mut second_anchors = anchor_components
        .iter()
        .copied()
        .filter(|&component_id| component_id != first_anchor)
        .collect::<Vec<_>>();
    second_anchors.sort_by(|&left, &right| {
        let opposite_split_score = |component_id: usize| {
            let candidate = member(component_id);
            (
                candidate.split_kind != first_member.split_kind,
                candidate.tie_rank == first_member.tie_rank,
            )
        };
        opposite_split_score(right)
            .cmp(&opposite_split_score(left))
            .then_with(|| current_h(right).total_cmp(&current_h(left)))
            .then_with(|| {
                member(left)
                    .abstract_operators
                    .cmp(&member(right).abstract_operators)
            })
            .then_with(|| left.cmp(&right))
    });
    let second_anchor = (!second_anchors.is_empty())
        .then(|| second_anchors[variant.second_anchor_round % second_anchors.len()]);

    let mut order = Vec::with_capacity(base_order.len());
    let mut selected = HashSet::with_capacity(base_order.len());
    if variant.non_goal_prefix {
        for &component_id in base_order {
            if !is_goal_member(component_id) {
                order.push(component_id);
                selected.insert(component_id);
            }
        }
    }
    order.push(first_anchor);
    selected.insert(first_anchor);
    if let Some(component_id) = second_anchor {
        order.push(component_id);
        selected.insert(component_id);
    }
    if progress_states {
        for component_id in sorted_anchor_components {
            if selected.insert(component_id) {
                order.push(component_id);
            }
        }
    }

    let mut other_goals = by_goal
        .iter()
        .filter(|(goal_id, _)| **goal_id != anchor_goal)
        .collect::<Vec<_>>();
    other_goals.sort_by(|(left_goal, left), (right_goal, right)| {
        goal_max_h(right)
            .total_cmp(&goal_max_h(left))
            .then_with(|| left_goal.cmp(right_goal))
    });
    for (representative_index, (_, components)) in other_goals.into_iter().enumerate() {
        let mut representatives = components.clone();
        representatives.sort_by(|&left, &right| {
            let variant_score = |component_id: usize| {
                let candidate = member(component_id);
                (
                    candidate.split_kind == first_member.split_kind,
                    candidate.tie_rank == first_member.tie_rank,
                )
            };
            variant_score(right)
                .cmp(&variant_score(left))
                .then_with(|| {
                    member(left)
                        .abstract_operators
                        .cmp(&member(right).abstract_operators)
                })
                .then_with(|| current_h(right).total_cmp(&current_h(left)))
                .then_with(|| left.cmp(&right))
        });
        let representative_offset = variant
            .representative_round
            .wrapping_add(representative_index.wrapping_mul(variant.anchor_offset + 1))
            % representatives.len();
        let representative = representatives[representative_offset];
        order.push(representative);
        selected.insert(representative);
    }

    if variant.compact {
        debug_assert_eq!(
            order.len(),
            if progress_states {
                by_goal.len() + anchor_components.len() - 1
            } else {
                by_goal.len() + 1
            }
        );
        return Some(order);
    }

    let mut remaining = base_order
        .iter()
        .copied()
        .filter(|component_id| !selected.contains(component_id))
        .collect::<Vec<_>>();
    remaining.sort_by(
        |&left, &right| match (is_goal_member(left), is_goal_member(right)) {
            (true, true) => member(left)
                .abstract_operators
                .cmp(&member(right).abstract_operators)
                .then_with(|| current_h(right).total_cmp(&current_h(left)))
                .then_with(|| left.cmp(&right)),
            (true, false) => std::cmp::Ordering::Less,
            (false, true) => std::cmp::Ordering::Greater,
            (false, false) => base_order
                .iter()
                .position(|&id| id == left)
                .cmp(&base_order.iter().position(|&id| id == right)),
        },
    );
    order.extend(remaining);
    debug_assert_eq!(order.len(), base_order.len());
    debug_assert_eq!(
        order.iter().copied().collect::<HashSet<_>>().len(),
        base_order.len()
    );
    Some(order)
}

pub(super) fn deduplicate_goal_cover_orders(
    orders: Vec<(Option<usize>, Vec<usize>)>,
) -> Vec<(Option<usize>, Vec<usize>)> {
    let mut index_by_order = HashMap::<Vec<usize>, usize>::with_capacity(orders.len());
    let mut unique = Vec::<(Option<usize>, Vec<usize>)>::with_capacity(orders.len());
    for (goal_cover_goal_id, order) in orders {
        if let Some(&index) = index_by_order.get(&order) {
            if unique[index].0.is_none() {
                unique[index].0 = goal_cover_goal_id;
            }
            continue;
        }
        index_by_order.insert(order.clone(), unique.len());
        unique.push((goal_cover_goal_id, order));
    }
    unique
}
