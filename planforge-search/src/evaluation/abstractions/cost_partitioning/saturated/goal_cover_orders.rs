use std::collections::{HashMap, HashSet};

use super::SaturatedCostPartitioningHeuristic;
use crate::evaluation::abstractions::common::cegar::SplitKind;
use crate::evaluation::abstractions::component::{AbstractionComponent, GoalFocusedMember};

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

/// A goal-cover order of `base_order` (AAMAS 2027): the anchors of one goal
/// first, then one representative of every other goal, so that the order
/// covers every goal early, and then, unless `variant` is compact, every
/// other component.
///
/// The goals are ranked by the largest estimate of their abstractions at the
/// evaluated state, `abstraction_h`, and `variant` picks the anchor goal, its
/// first and second anchor and the representatives by offsets into those
/// rankings. Without progress states, only goals with at least two
/// abstractions take part. `None` when no goal does, or when a compact
/// variant or `require_pure_goal_focused_collection` meets components outside
/// a goal-focused collection.
pub(super) fn goal_cover_order(
    base_order: &[usize],
    components: &[AbstractionComponent<'_>],
    abstraction_h: &[f64],
    require_pure_goal_focused_collection: bool,
    variant: GoalCoverOrderVariant,
) -> Option<Vec<usize>> {
    let ranking = GoalCoverRanking {
        components,
        abstraction_h,
    };
    let pure_goal_focused_collection = base_order.iter().all(|&id| ranking.is_goal_member(id));
    if (require_pure_goal_focused_collection || variant.compact) && !pure_goal_focused_collection {
        return None;
    }

    let mut by_goal: HashMap<usize, Vec<usize>> = HashMap::new();
    for &component_id in base_order {
        if let Some(member) = components[component_id].goal_focused_member() {
            by_goal
                .entry(member.membership.goal_id)
                .or_default()
                .push(component_id);
        }
    }
    let progress_states = components
        .iter()
        .filter_map(AbstractionComponent::goal_focused_member)
        .any(|member| member.membership.from_progress_state);
    let anchor_goals = ranking.goals_by_max_h(&by_goal, |components| {
        progress_states || components.len() >= 2
    });
    if anchor_goals.is_empty() {
        return None;
    }
    let (anchor_goal, anchor_components) =
        anchor_goals[variant.anchor_goal_offset % anchor_goals.len()];
    let anchors = ranking.anchors(anchor_components, variant);

    let mut order = Vec::with_capacity(base_order.len());
    let mut selected = HashSet::with_capacity(base_order.len());
    let mut select = |component_id: usize, order: &mut Vec<usize>| {
        if selected.insert(component_id) {
            order.push(component_id);
        }
    };
    if variant.non_goal_prefix {
        for &component_id in base_order {
            if !ranking.is_goal_member(component_id) {
                select(component_id, &mut order);
            }
        }
    }
    select(anchors.first, &mut order);
    if let Some(second) = anchors.second {
        select(second, &mut order);
    }
    if progress_states {
        for &component_id in &anchors.ranked {
            select(component_id, &mut order);
        }
    }
    let first_member = ranking.member(anchors.first);
    let other_goals = ranking.goals_by_max_h(&by_goal, |_| true);
    for (representative_index, (_, components)) in other_goals
        .into_iter()
        .filter(|&(goal_id, _)| goal_id != anchor_goal)
        .enumerate()
    {
        let representative =
            ranking.representative(components, first_member, representative_index, variant);
        select(representative, &mut order);
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

    // The rest: goal-focused members by size, then the other components in
    // the base order (the sort is stable).
    let mut remaining = base_order
        .iter()
        .copied()
        .filter(|component_id| !selected.contains(component_id))
        .collect::<Vec<_>>();
    remaining.sort_by(|&left, &right| {
        match (ranking.is_goal_member(left), ranking.is_goal_member(right)) {
            (true, true) => ranking
                .member(left)
                .abstract_operators
                .cmp(&ranking.member(right).abstract_operators)
                .then_with(|| ranking.h(right).total_cmp(&ranking.h(left)))
                .then_with(|| left.cmp(&right)),
            (true, false) => std::cmp::Ordering::Less,
            (false, true) => std::cmp::Ordering::Greater,
            (false, false) => std::cmp::Ordering::Equal,
        }
    });
    order.extend(remaining);
    debug_assert_eq!(order.len(), base_order.len());
    Some(order)
}

/// The two anchors of the anchor goal and all its components ranked.
struct Anchors {
    first: usize,
    second: Option<usize>,
    ranked: Vec<usize>,
}

/// What goal-cover orders rank components by: their estimates at the
/// evaluated state and their goal-focused membership.
struct GoalCoverRanking<'a, 'task> {
    components: &'a [AbstractionComponent<'task>],
    abstraction_h: &'a [f64],
}

impl GoalCoverRanking<'_, '_> {
    fn h(&self, component_id: usize) -> f64 {
        self.abstraction_h[component_id]
    }

    fn is_goal_member(&self, component_id: usize) -> bool {
        self.components[component_id]
            .goal_focused_member()
            .is_some()
    }

    fn member(&self, component_id: usize) -> GoalFocusedMember {
        self.components[component_id]
            .goal_focused_member()
            .expect("goal-cover order component must reference a goal-focused collection member")
    }

    /// The goals of `by_goal` that `admit` their components, by the largest
    /// estimate among their components, descending, then by goal id.
    fn goals_by_max_h<'g>(
        &self,
        by_goal: &'g HashMap<usize, Vec<usize>>,
        admit: impl Fn(&[usize]) -> bool,
    ) -> Vec<(usize, &'g [usize])> {
        let goal_max_h = |components: &[usize]| {
            components
                .iter()
                .map(|&component_id| self.h(component_id))
                .fold(0.0, f64::max)
        };
        let mut goals = by_goal
            .iter()
            .filter(|(_, components)| admit(components))
            .map(|(&goal_id, components)| (goal_id, components.as_slice()))
            .collect::<Vec<_>>();
        goals.sort_by(|(left_goal, left), (right_goal, right)| {
            goal_max_h(right)
                .total_cmp(&goal_max_h(left))
                .then_with(|| left_goal.cmp(right_goal))
        });
        goals
    }

    /// The anchors of the anchor goal: `variant`'s pick among its components
    /// by estimate, preferring later tie ranks and wanted splits, and a second
    /// one that differs from the first in split kind and tie rank, preferring
    /// high estimates and few abstract operators.
    fn anchors(&self, anchor_components: &[usize], variant: GoalCoverOrderVariant) -> Anchors {
        let mut ranked = anchor_components.to_vec();
        ranked.sort_by(|&left, &right| {
            let (left_member, right_member) = (self.member(left), self.member(right));
            self.h(right)
                .total_cmp(&self.h(left))
                .then_with(|| right_member.tie_rank.cmp(&left_member.tie_rank))
                .then_with(|| {
                    (right_member.split_kind == SplitKind::Wanted)
                        .cmp(&(left_member.split_kind == SplitKind::Wanted))
                })
                .then_with(|| left.cmp(&right))
        });
        let first = ranked[variant.anchor_offset % ranked.len()];
        let first_member = self.member(first);
        let opposite_split_score = |component_id: usize| {
            let candidate = self.member(component_id);
            (
                candidate.split_kind != first_member.split_kind,
                candidate.tie_rank == first_member.tie_rank,
            )
        };
        let mut seconds = anchor_components
            .iter()
            .copied()
            .filter(|&component_id| component_id != first)
            .collect::<Vec<_>>();
        seconds.sort_by(|&left, &right| {
            opposite_split_score(right)
                .cmp(&opposite_split_score(left))
                .then_with(|| self.h(right).total_cmp(&self.h(left)))
                .then_with(|| {
                    self.member(left)
                        .abstract_operators
                        .cmp(&self.member(right).abstract_operators)
                })
                .then_with(|| left.cmp(&right))
        });
        let second =
            (!seconds.is_empty()).then(|| seconds[variant.second_anchor_round % seconds.len()]);
        Anchors {
            first,
            second,
            ranked,
        }
    }

    /// The representative of the goal with `components`, the
    /// `representative_index`-th goal after the anchor goal: `variant`'s pick
    /// among them, preferring the first anchor's split kind and tie rank, few
    /// abstract operators and high estimates.
    fn representative(
        &self,
        components: &[usize],
        first_member: GoalFocusedMember,
        representative_index: usize,
        variant: GoalCoverOrderVariant,
    ) -> usize {
        let variant_score = |component_id: usize| {
            let candidate = self.member(component_id);
            (
                candidate.split_kind == first_member.split_kind,
                candidate.tie_rank == first_member.tie_rank,
            )
        };
        let mut representatives = components.to_vec();
        representatives.sort_by(|&left, &right| {
            variant_score(right)
                .cmp(&variant_score(left))
                .then_with(|| {
                    self.member(left)
                        .abstract_operators
                        .cmp(&self.member(right).abstract_operators)
                })
                .then_with(|| self.h(right).total_cmp(&self.h(left)))
                .then_with(|| left.cmp(&right))
        });
        let offset = variant
            .representative_round
            .wrapping_add(representative_index.wrapping_mul(variant.anchor_offset + 1))
            % representatives.len();
        representatives[offset]
    }
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
