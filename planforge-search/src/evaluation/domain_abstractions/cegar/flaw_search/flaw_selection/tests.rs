use std::collections::HashSet;

use planforge_sas::axioms::{ComparisonAxiom, ComparisonOperator};
use planforge_sas::numeric_task::{
    ExplicitFact, NumericType, NumericVariable, VariableIndex, ZERO_VALUE,
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
