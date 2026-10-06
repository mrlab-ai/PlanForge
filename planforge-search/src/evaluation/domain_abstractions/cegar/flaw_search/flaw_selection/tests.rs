use std::collections::HashSet;

use planforge_sas::axioms::{ComparisonAxiom, ComparisonOperator};
use planforge_sas::numeric_task::{
    ExplicitFact, ExplicitValueIndex, NumericType, NumericValue, NumericVariable, VariableIndex,
    ZERO_VALUE,
};
use rand::SeedableRng;

use super::*;
use crate::evaluation::domain_abstractions::cegar::flaw_search::PropFlaw;

/// One numeric condition `x0 = x0` writing propositional variable
/// `prop_var_id`; only the "is this a condition variable?" answer matters
/// for flaw scoring.
fn condition_on_prop_var(prop_var_id: VariableIndex) -> NumericConditions {
    NumericConditions::build(
        prop_var_id.index() + 1,
        &[NumericVariable::new(
            "x0".into(),
            NumericType::Regular,
            None,
        )],
        &[ComparisonAxiom::new(
            prop_var_id,
            VariableIndex::from_usize(0),
            VariableIndex::from_usize(0),
            ComparisonOperator::Equal,
        )],
        &[],
    )
    .unwrap()
}

fn prop_flaw(var: VariableIndex, deps: Vec<NumericFlaw>) -> Flaw {
    Flaw::Propositional(PropFlaw {
        fact: ExplicitFact::propositional(var.index(), 0),
        dependent_numeric_flaws: deps,
        step: 0,
    })
}

fn numeric_flaw(var: VariableIndex) -> NumericFlaw {
    NumericFlaw {
        numeric_var_id: var,
        value: ZERO_VALUE,
        include_in_lower: true,
        step: 0,
    }
}

#[test]
fn max_refined_scores_comparison_flaws_like_numeric_fd() {
    let flaws = vec![
        prop_flaw(VariableIndex::from_usize(1), Vec::new()),
        prop_flaw(
            VariableIndex::from_usize(0),
            vec![numeric_flaw(VariableIndex::from_usize(0))],
        ),
    ];
    let conditions = condition_on_prop_var(VariableIndex::from_usize(0));
    let domain_sizes = vec![1, 2];
    let numeric_domain_sizes = vec![1];
    let mut rng = SmallRng::seed_from_u64(1);

    let chosen = fix_single_flaw_max_refined(
        &flaws,
        &conditions,
        &domain_sizes,
        &numeric_domain_sizes,
        1,
        &mut rng,
    );

    assert_eq!(chosen[0].idx, 0);
}

#[test]
fn max_refined_continues_most_refined_dependent_numeric_view() {
    let flaws = vec![prop_flaw(
        VariableIndex::from_usize(0),
        vec![
            numeric_flaw(VariableIndex::from_usize(0)),
            numeric_flaw(VariableIndex::from_usize(1)),
        ],
    )];
    let conditions = condition_on_prop_var(VariableIndex::from_usize(0));
    let domain_sizes = vec![2];
    let numeric_domain_sizes = vec![7, 2];

    let (chosen, _) =
        compute_max_refined(&flaws, &conditions, &domain_sizes, &numeric_domain_sizes, 1);

    let restricted = chosen[0]
        .restricted_dep
        .as_ref()
        .expect("comparison flaw should restrict dependent numeric flaws");
    assert_eq!(
        restricted,
        &vec![numeric_flaw(VariableIndex::from_usize(0))]
    );
}

#[test]
fn min_growth_continues_most_refined_dependent_numeric_view() {
    let flaws = vec![prop_flaw(
        VariableIndex::from_usize(0),
        vec![
            numeric_flaw(VariableIndex::from_usize(0)),
            numeric_flaw(VariableIndex::from_usize(1)),
        ],
    )];
    let conditions = condition_on_prop_var(VariableIndex::from_usize(0));
    let domain_sizes = vec![2];
    let numeric_domain_sizes = vec![7, 2];
    let mut rng = SmallRng::seed_from_u64(1);

    let chosen = fix_single_flaw_min_growth(
        &flaws,
        &conditions,
        &domain_sizes,
        &numeric_domain_sizes,
        &mut rng,
    );

    let restricted = chosen[0]
        .restricted_dep
        .as_ref()
        .expect("comparison flaw should restrict dependent numeric flaws");
    assert_eq!(
        restricted,
        &vec![numeric_flaw(VariableIndex::from_usize(0))]
    );
}

#[test]
fn min_growth_randomizes_equal_growth_candidates_across_seeds() {
    let flaws = vec![
        Flaw::Numeric(numeric_flaw(VariableIndex::from_usize(0))),
        Flaw::Numeric(numeric_flaw(VariableIndex::from_usize(1))),
        Flaw::Numeric(numeric_flaw(VariableIndex::from_usize(2))),
    ];
    let conditions = NumericConditions::default();
    let mut first_choices = HashSet::new();

    for seed in 0..32 {
        let domain_sizes = Vec::new();
        let numeric_domain_sizes = vec![1, 1, 1];
        let mut rng = SmallRng::seed_from_u64(seed);
        let chosen = fix_single_flaw_min_growth(
            &flaws,
            &conditions,
            &domain_sizes,
            &numeric_domain_sizes,
            &mut rng,
        );
        first_choices.insert(chosen[0].idx);
    }

    assert_eq!(first_choices, HashSet::from([0, 1, 2]));
}

fn numeric_flaw_on(var: usize, value: f64) -> Flaw {
    Flaw::Numeric(NumericFlaw {
        numeric_var_id: VariableIndex::from_usize(var),
        value: NumericValue::new(value),
        include_in_lower: true,
        step: 0,
    })
}

fn first_choice(
    treatment: FlawTreatmentVariants,
    flaws: &[Flaw],
    numeric_domain_sizes: &[usize],
    seed: u64,
    rank: Option<usize>,
) -> usize {
    let task = planforge_sas::numeric_task::NumericRootTask::new(
        planforge_sas::numeric_task::NumericRootTaskParts {
            version: 1,
            metric: planforge_sas::numeric_task::Metric::new(true, None),
            variables: vec![planforge_sas::numeric_task::ExplicitVariable::new(
                2,
                "v".into(),
                vec!["a".into(), "b".into()],
                None,
                ExplicitValueIndex::new(0),
            )],
            numeric_variables: vec![],
            goals: vec![ExplicitFact::propositional(0, 1)],
            mutexes: vec![],
            state: vec![ExplicitValueIndex::new(0)],
            numeric_state: vec![],
            operators: vec![],
            axioms: vec![],
            comparison_axioms: vec![],
            assignment_axioms: vec![],
            global_constraint: ExplicitFact::propositional(0, 0),
        },
    );
    let mut rng = SmallRng::seed_from_u64(seed);
    let tie_break = match rank {
        Some(rank) => TieBreak::Rank(rank),
        None => TieBreak::Random(&mut rng),
    };
    let chosen = treatment.choose_flaws(&task, flaws, &[1], numeric_domain_sizes, tie_break, 1);
    assert_eq!(chosen.len(), flaws.len(), "every flaw stays a fallback");
    chosen[0].idx
}

#[test]
fn ranked_min_growth_picks_among_the_tied_best_by_a_deterministic_key() {
    let flaws = vec![
        numeric_flaw_on(3, 0.0),
        numeric_flaw_on(1, 0.0),
        numeric_flaw_on(0, 0.0),
        numeric_flaw_on(2, 0.0),
    ];
    // A split grows a dimension of n partitions by (n + 1) / n: variable 0
    // (one partition) by 2, variables 1, 2, 3 (three each) by 4/3, so the
    // latter tie for the best and variable 0 is never ranked.
    let numeric_domain_sizes = [1, 3, 3, 3];
    for seed in 0..8 {
        let pick = |rank| {
            first_choice(
                FlawTreatmentVariants::MinGrowthSingleAtom,
                &flaws,
                &numeric_domain_sizes,
                seed,
                Some(rank),
            )
        };
        // The ties sorted by variable are 1, 2, 3 at flaw indices 1, 3, 0.
        assert_eq!(pick(0), 1);
        assert_eq!(pick(1), 3);
        assert_eq!(pick(2), 0);
        assert_eq!(pick(3), 1, "ranks wrap around the tie");
        assert_eq!(pick(5), 0);
    }
}

#[test]
fn ranked_selection_ignores_the_input_order_and_the_seed() {
    let flaws = vec![
        numeric_flaw_on(2, 1.0),
        numeric_flaw_on(0, 1.0),
        numeric_flaw_on(1, 1.0),
    ];
    let reversed = flaws.iter().rev().cloned().collect::<Vec<_>>();
    for treatment in [
        FlawTreatmentVariants::RandomSingleAtom,
        FlawTreatmentVariants::MaxRefinedSingleAtom,
        FlawTreatmentVariants::MinGrowthSingleAtom,
        FlawTreatmentVariants::ClosestToGoal,
    ] {
        for rank in 0..3 {
            let choices = (0..8)
                .flat_map(|seed| {
                    [
                        flaws[first_choice(treatment, &flaws, &[1, 1, 1], seed, Some(rank))]
                            .clone(),
                        reversed[first_choice(treatment, &reversed, &[1, 1, 1], seed, Some(rank))]
                            .clone(),
                    ]
                })
                .collect::<Vec<_>>();
            assert!(
                choices
                    .iter()
                    .all(|choice| *choice == numeric_flaw_on(rank, 1.0)),
                "{treatment} rank {rank}: {choices:?}"
            );
        }
    }
}

#[test]
fn refine_all_treatments_have_no_tie_to_rank() {
    let flaws = vec![numeric_flaw_on(1, 0.0), numeric_flaw_on(0, 0.0)];
    for treatment in [
        FlawTreatmentVariants::OneSplitPerAtom,
        FlawTreatmentVariants::OneSplitPerVariable,
    ] {
        assert_eq!(
            first_choice(treatment, &flaws, &[1, 1], 0, Some(1)),
            first_choice(treatment, &flaws, &[1, 1], 0, None),
        );
    }
}
