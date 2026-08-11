use crate::axioms::{ComparisonAxiom, ComparisonOperator};
use crate::numeric_conditions::ConditionValue;
use crate::numeric_task::{
    ExplicitVariable, Metric, NumericRootTask, NumericRootTaskParts, NumericType, NumericVariable,
    Operator,
};

use super::*;

/// A binary non-derived variable. Value 0 is the atom, 1 its absence.
fn plain(name: &str) -> ExplicitVariable {
    sized(name, 2)
}

fn sized(name: &str, domain_size: usize) -> ExplicitVariable {
    ExplicitVariable::new(
        domain_size,
        name.to_string(),
        (0..domain_size)
            .map(|value| format!("{name}={value}"))
            .collect(),
        None,
        0,
    )
}

/// A binary derived variable at `layer`, defaulting to 1 (the atom absent).
fn derived(name: &str, layer: usize) -> ExplicitVariable {
    ExplicitVariable::new(
        2,
        name.to_string(),
        vec![format!("{name}"), format!("not {name}")],
        Some(layer),
        1,
    )
}

fn condition_variable(name: &str, layer: usize) -> ExplicitVariable {
    ExplicitVariable::new(
        ConditionValue::DOMAIN_SIZE,
        name.to_string(),
        vec![format!("{name}"), format!("not {name}")],
        Some(layer),
        ConditionValue::False.as_usize(),
    )
}

/// `head=0 <- body`, i.e. a rule proving a derived variable that defaults to 1.
fn proves(head: usize, body: Vec<ExplicitFact>) -> PropositionalAxiom {
    PropositionalAxiom::new(body, head, 1, 0)
}

struct TaskBuilder {
    variables: Vec<ExplicitVariable>,
    axioms: Vec<PropositionalAxiom>,
    goals: Vec<ExplicitFact>,
    operators: Vec<Operator>,
    numeric_variables: Vec<NumericVariable>,
    numeric_state: Vec<f64>,
    comparison_axioms: Vec<ComparisonAxiom>,
}

impl TaskBuilder {
    fn new(variables: Vec<ExplicitVariable>) -> Self {
        TaskBuilder {
            variables,
            axioms: Vec::new(),
            goals: Vec::new(),
            operators: Vec::new(),
            numeric_variables: Vec::new(),
            numeric_state: Vec::new(),
            comparison_axioms: Vec::new(),
        }
    }

    fn axioms(mut self, axioms: Vec<PropositionalAxiom>) -> Self {
        self.axioms = axioms;
        self
    }

    fn goals(mut self, goals: Vec<ExplicitFact>) -> Self {
        self.goals = goals;
        self
    }

    fn operator(mut self, preconditions: Vec<ExplicitFact>) -> Self {
        self.operators.push(Operator::new(
            format!("op{}", self.operators.len()),
            preconditions,
            vec![],
            vec![],
            1,
        ));
        self
    }

    /// A comparison axiom over two constants, so the verdict is fixed and the
    /// initial-state closure has something well-defined to compute.
    fn comparison(mut self, affected: usize) -> Self {
        self.numeric_variables = vec![
            NumericVariable::new("left".to_string(), NumericType::Constant, None),
            NumericVariable::new("right".to_string(), NumericType::Constant, None),
        ];
        self.numeric_state = vec![1.0, 5.0];
        self.comparison_axioms.push(ComparisonAxiom::new(
            affected,
            0,
            1,
            ComparisonOperator::GreaterThanOrEqual,
        ));
        self
    }

    fn build(self) -> NumericRootTask {
        let state = self
            .variables
            .iter()
            .map(|variable| variable.domain_size() - 1)
            .collect();
        NumericRootTask::new(NumericRootTaskParts {
            version: 1,
            metric: Metric::new(true, None),
            variables: self.variables,
            numeric_variables: self.numeric_variables,
            goals: self.goals,
            mutexes: vec![],
            state,
            numeric_state: self.numeric_state,
            operators: self.operators,
            axioms: self.axioms,
            comparison_axioms: self.comparison_axioms,
            assignment_axioms: vec![],
            global_constraint: ExplicitFact::propositional(0, 0),
        })
    }
}

/// One produced rule as `(head variable, head value, sorted body)`.
type Rule = (usize, usize, Vec<(usize, usize)>);

/// The produced rules, in the order [`default_value_axioms`] returns them.
fn rules(task: &NumericRootTask, mode: DefaultValueAxiomMode) -> Vec<Rule> {
    default_value_axioms(task, mode)
        .iter()
        .map(|axiom| {
            let mut body: Vec<(usize, usize)> = axiom
                .conditions()
                .iter()
                .map(|fact| (fact.var(), fact.value()))
                .collect();
            body.sort_unstable();
            (axiom.var_id(), axiom.effect_value(), body)
        })
        .collect()
}

/// One conjunctive rule negates into one rule per body literal.
#[test]
fn a_conjunctive_body_negates_into_one_rule_per_literal() {
    let task = TaskBuilder::new(vec![plain("a"), plain("b"), derived("d", 0)])
        .axioms(vec![proves(
            2,
            vec![
                ExplicitFact::propositional(0, 0),
                ExplicitFact::propositional(1, 1),
            ],
        )])
        .goals(vec![ExplicitFact::propositional(2, 1)])
        .build();

    assert_eq!(
        rules(&task, DefaultValueAxiomMode::ApproximateNegativeCycles),
        vec![(2, 1, vec![(0, 1)]), (2, 1, vec![(1, 0)]),]
    );
}

/// Two rules proving the same variable negate into their cross product, which
/// for one literal each is a single rule needing both to fail.
#[test]
fn disjunctive_support_negates_into_a_conjunction() {
    let task = TaskBuilder::new(vec![plain("a"), plain("b"), derived("d", 0)])
        .axioms(vec![
            proves(2, vec![ExplicitFact::propositional(0, 0)]),
            proves(2, vec![ExplicitFact::propositional(1, 0)]),
        ])
        .goals(vec![ExplicitFact::propositional(2, 1)])
        .build();

    assert_eq!(
        rules(&task, DefaultValueAxiomMode::ApproximateNegativeCycles),
        vec![(2, 1, vec![(0, 1), (1, 1)])]
    );
}

/// A body condition on a multi-valued variable negates into a disjunction over
/// the variable's other values, which becomes one rule per value.
///
/// This is the shape the translator could not produce: it negated PDDL literals,
/// where "not this atom" is a single literal, whereas after the SAS encoding the
/// same condition is one value of a three-valued variable.
#[test]
fn a_multi_valued_condition_negates_into_a_disjunction() {
    let task = TaskBuilder::new(vec![sized("a", 3), derived("d", 0)])
        .axioms(vec![proves(1, vec![ExplicitFact::propositional(0, 0)])])
        .goals(vec![ExplicitFact::propositional(1, 1)])
        .build();

    assert_eq!(
        rules(&task, DefaultValueAxiomMode::ApproximateNegativeCycles),
        vec![(1, 1, vec![(0, 1)]), (1, 1, vec![(0, 2)])]
    );
}

/// A hitting set with a fact that no clause needs it alone for is dropped in
/// favour of the subset that still hits everything.
#[test]
fn a_dominated_hitting_set_is_dropped() {
    // `d <- a=0` and `d <- a=0 ∧ b=0`. The second rule is only reachable when
    // the first is, so refuting `d` needs `a=1` and nothing else; `{a=1, b=1}`
    // hits both clauses but `b=1` is never the sole reason.
    let task = TaskBuilder::new(vec![plain("a"), plain("b"), derived("d", 0)])
        .axioms(vec![
            proves(2, vec![ExplicitFact::propositional(0, 0)]),
            proves(
                2,
                vec![
                    ExplicitFact::propositional(0, 0),
                    ExplicitFact::propositional(1, 0),
                ],
            ),
        ])
        .goals(vec![ExplicitFact::propositional(2, 1)])
        .build();

    assert_eq!(
        rules(&task, DefaultValueAxiomMode::ApproximateNegativeCycles),
        vec![(2, 1, vec![(0, 1)])]
    );
}

/// Nothing refutes a variable that holds unconditionally.
///
/// This is what keeps the translator's global-constraint atom inert: it is proven
/// by an empty body, so even a consumer that observed it would get no rule.
#[test]
fn an_unconditionally_proven_variable_gets_no_rule() {
    let task = TaskBuilder::new(vec![derived("always", 0)])
        .axioms(vec![proves(0, vec![])])
        .goals(vec![ExplicitFact::propositional(0, 1)])
        .build();

    assert!(
        default_value_axioms(&task, DefaultValueAxiomMode::ApproximateNegativeCycles).is_empty()
    );
}

/// A variable whose default value nothing observes gets no rule.
#[test]
fn a_variable_read_only_at_its_nondefault_value_gets_no_rule() {
    let task = TaskBuilder::new(vec![plain("a"), derived("d", 0)])
        .axioms(vec![proves(1, vec![ExplicitFact::propositional(0, 0)])])
        // The goal asks for `d` proven, never for `d` refuted.
        .goals(vec![ExplicitFact::propositional(1, 0)])
        .build();

    assert!(
        default_value_axioms(&task, DefaultValueAxiomMode::ApproximateNegativeCycles).is_empty()
    );
}

/// An operator precondition reading a derived variable at its default value is
/// an observation, exactly like a goal.
#[test]
fn an_operator_precondition_makes_a_default_value_relevant() {
    let task = TaskBuilder::new(vec![plain("a"), derived("d", 0)])
        .axioms(vec![proves(1, vec![ExplicitFact::propositional(0, 0)])])
        .operator(vec![ExplicitFact::propositional(1, 1)])
        .build();

    assert_eq!(
        rules(&task, DefaultValueAxiomMode::ApproximateNegativeCycles),
        vec![(1, 1, vec![(0, 1)])]
    );
}

/// Observing a default value transitively requires the values its rules read.
///
/// `high` is observed refuted; refuting it needs `low` refuted, and `low` is
/// observed nowhere else, so without the propagation it would get no rule and a
/// heuristic would find `high` unrefutable.
#[test]
fn relevance_propagates_through_the_rules_of_an_observed_variable() {
    let task = TaskBuilder::new(vec![plain("a"), derived("low", 0), derived("high", 0)])
        .axioms(vec![
            proves(1, vec![ExplicitFact::propositional(0, 0)]),
            proves(2, vec![ExplicitFact::propositional(1, 0)]),
        ])
        .goals(vec![ExplicitFact::propositional(2, 1)])
        .build();

    assert_eq!(
        rules(&task, DefaultValueAxiomMode::ApproximateNegativeCycles),
        vec![(1, 1, vec![(0, 1)]), (2, 1, vec![(1, 1)])]
    );
}

/// A cyclic nondefault dependency is refuted unconditionally rather than
/// literal by literal, which would claim the variables can never be refuted.
///
/// The empty body is also where the relevance analysis stops: `p=default` no
/// longer depends on `q` at all, so observing only `p` does not drag `q` in even
/// though `p`'s proving rule reads it.
#[test]
fn a_cyclic_component_is_refuted_unconditionally() {
    let cycle = |goals: Vec<ExplicitFact>| {
        TaskBuilder::new(vec![plain("a"), derived("p", 0), derived("q", 0)])
            .axioms(vec![
                proves(
                    1,
                    vec![
                        ExplicitFact::propositional(0, 0),
                        ExplicitFact::propositional(2, 0),
                    ],
                ),
                proves(2, vec![ExplicitFact::propositional(1, 0)]),
            ])
            .goals(goals)
            .build()
    };

    let both_observed = cycle(vec![
        ExplicitFact::propositional(1, 1),
        ExplicitFact::propositional(2, 1),
    ]);
    assert_eq!(
        rules(
            &both_observed,
            DefaultValueAxiomMode::ApproximateNegativeCycles
        ),
        vec![(1, 1, vec![]), (2, 1, vec![])]
    );

    let one_observed = cycle(vec![ExplicitFact::propositional(1, 1)]);
    assert_eq!(
        rules(
            &one_observed,
            DefaultValueAxiomMode::ApproximateNegativeCycles
        ),
        vec![(1, 1, vec![])]
    );
}

/// The trivial mode refutes everything relevant unconditionally, cycle or not.
#[test]
fn the_trivial_mode_refutes_every_relevant_variable_unconditionally() {
    let task = TaskBuilder::new(vec![plain("a"), plain("b"), derived("d", 0)])
        .axioms(vec![proves(
            2,
            vec![
                ExplicitFact::propositional(0, 0),
                ExplicitFact::propositional(1, 1),
            ],
        )])
        .goals(vec![ExplicitFact::propositional(2, 1)])
        .build();

    assert_eq!(
        rules(&task, DefaultValueAxiomMode::ApproximateNegative),
        vec![(2, 1, vec![])]
    );
}

/// A condition variable is not derived: it is computed by the comparison pass,
/// so nothing has to explain how it becomes false — but a rule reading it *is*
/// negated over its two values like any other variable.
#[test]
fn a_condition_variable_needs_no_rule_but_is_negated_into_one() {
    let task = TaskBuilder::new(vec![
        plain("a"),
        condition_variable("cmp", 0),
        derived("d", 1),
    ])
    .comparison(1)
    .axioms(vec![proves(
        2,
        vec![ExplicitFact::propositional(
            1,
            ConditionValue::True.as_usize(),
        )],
    )])
    .goals(vec![ExplicitFact::propositional(2, 1)])
    .build();

    assert_eq!(
        rules(&task, DefaultValueAxiomMode::ApproximateNegativeCycles),
        vec![(2, 1, vec![(1, ConditionValue::False.as_usize())])]
    );
}

/// The rules are tagged in the same fact namespaces the task uses, so a consumer
/// that distinguishes a condition variable from a propositional one still can.
#[test]
fn rule_bodies_carry_the_task_fact_namespaces() {
    let task = TaskBuilder::new(vec![
        plain("a"),
        condition_variable("cmp", 0),
        derived("d", 1),
    ])
    .comparison(1)
    .axioms(vec![proves(
        2,
        vec![
            ExplicitFact::propositional(0, 0),
            ExplicitFact::propositional(1, ConditionValue::True.as_usize()),
        ],
    )])
    .goals(vec![ExplicitFact::propositional(2, 1)])
    .build();

    let produced = default_value_axioms(&task, DefaultValueAxiomMode::ApproximateNegativeCycles);
    let namespaces: Vec<Vec<bool>> = produced
        .iter()
        .map(|axiom| {
            axiom
                .conditions()
                .iter()
                .map(ExplicitFact::is_condition)
                .collect()
        })
        .collect();
    assert_eq!(namespaces, vec![vec![false], vec![true]]);
}
