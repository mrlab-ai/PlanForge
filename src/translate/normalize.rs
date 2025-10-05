use crate::translate::build_model as bm;
use crate::translate::pddl_ast::Condition;
use std::collections::{HashMap, HashSet};

const OBJECT_PREDICATE: &str = "@object";

#[derive(Default, Debug, PartialEq)]
pub struct NormalizationOutcome {
    pub new_facts: Vec<bm::Atom>,
    pub object_predicate_required: bool,
}

fn collect_condition_vars(rule: &bm::RuleSpec) -> HashSet<String> {
    let mut vars = HashSet::new();
    for cond in &rule.conditions {
        for arg in &cond.args {
            if arg.starts_with('?') {
                vars.insert(arg.clone());
            }
        }
    }
    vars
}

/// Adds `@object(?x)` conditions for variables that only occur in the rule head.
fn remove_free_effect_variables(rules: &mut [bm::RuleSpec]) -> bool {
    let mut inserted = false;
    for rule in rules.iter_mut() {
        let mut bound_vars = collect_condition_vars(rule);
        let mut extra_conditions: Vec<bm::SymAtom> = Vec::new();
        for arg in &rule.effect.args {
            if !arg.starts_with('?') {
                continue;
            }
            if bound_vars.contains(arg) {
                continue;
            }
            let already_present = rule
                .conditions
                .iter()
                .any(|c| c.predicate == OBJECT_PREDICATE && c.args.len() == 1 && &c.args[0] == arg);
            if !already_present {
                extra_conditions.push(bm::SymAtom::new(
                    OBJECT_PREDICATE.to_string(),
                    vec![arg.clone()],
                ));
            }
            bound_vars.insert(arg.clone());
        }
        if !extra_conditions.is_empty() {
            rule.conditions.extend(extra_conditions);
            inserted = true;
        }
    }
    inserted
}

/// Deduplicates identical conditions within each rule to keep the join queue small.
fn split_duplicate_arguments(rules: &mut [bm::RuleSpec]) {
    for rule in rules.iter_mut() {
        let mut seen: HashSet<(String, Vec<String>)> = HashSet::new();
        rule.conditions.retain(|cond| {
            let key = (cond.predicate.clone(), cond.args.clone());
            seen.insert(key)
        });
    }
}

/// Converts rules without conditions (and constant heads) into base facts.
fn convert_trivial_rules(rules: &mut Vec<bm::RuleSpec>) -> Vec<bm::Atom> {
    let mut produced: Vec<bm::Atom> = Vec::new();
    rules.retain(|rule| {
        if rule.conditions.is_empty() {
            if rule.effect.args.iter().any(|a| a.starts_with('?')) {
                // Unable to convert – keep the rule around for later processing.
                true
            } else {
                produced.push(bm::Atom {
                    predicate: rule.effect.predicate.clone(),
                    args: rule
                        .effect
                        .args
                        .iter()
                        .map(|a| bm::Arg::Const(a.clone()))
                        .collect(),
                });
                false
            }
        } else {
            true
        }
    });
    produced
}

pub fn normalize_rules(rules: &mut Vec<bm::RuleSpec>) -> NormalizationOutcome {
    let mut outcome = NormalizationOutcome::default();
    outcome.object_predicate_required = remove_free_effect_variables(rules);
    split_duplicate_arguments(rules);
    outcome.new_facts = convert_trivial_rules(rules);
    outcome
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rule(effect: (&str, Vec<&str>), conds: Vec<(&str, Vec<&str>)>, rtype: &str) -> bm::RuleSpec {
        bm::RuleSpec {
            rtype: rtype.to_string(),
            effect: sym_atom(effect.0, effect.1),
            conditions: conds
                .into_iter()
                .map(|(p, args)| sym_atom(p, args))
                .collect(),
        }
    }

    fn sym_atom(pred: &str, args: Vec<&str>) -> bm::SymAtom {
        bm::SymAtom::new(
            pred.to_string(),
            args.into_iter().map(|a| a.to_string()).collect(),
        )
    }

    #[test]
    fn adds_object_condition_for_free_head_var() {
        let mut rules = vec![rule(("move", vec!["?x"]), vec![], "project")];
        let inserted = remove_free_effect_variables(&mut rules);
        assert!(inserted);
        assert_eq!(rules[0].conditions.len(), 1);
        assert_eq!(rules[0].conditions[0].predicate, OBJECT_PREDICATE);
        assert_eq!(rules[0].conditions[0].args, vec!["?x".to_string()]);
    }

    #[test]
    fn duplicate_conditions_are_removed() {
        let mut rules = vec![rule(
            ("move", vec!["?x"]),
            vec![("at", vec!["?x"]), ("at", vec!["?x"])],
            "project",
        )];
        split_duplicate_arguments(&mut rules);
        assert_eq!(rules[0].conditions.len(), 1);
    }

    #[test]
    fn trivial_constant_rule_becomes_fact() {
        let mut rules = vec![rule(("ready", vec!["a1"]), vec![], "project")];
        let facts = convert_trivial_rules(&mut rules);
        assert!(rules.is_empty());
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].predicate, "ready");
        assert_eq!(facts[0].args, vec![bm::Arg::Const("a1".to_string())]);
    }

    #[test]
    fn normalization_pipeline_runs_steps() {
        let mut rules = vec![rule(("move", vec!["?x"]), vec![], "project")];
        let outcome = normalize_rules(&mut rules);
        assert!(outcome.object_predicate_required);
        assert!(outcome.new_facts.is_empty());
        assert_eq!(rules.len(), 1);
        assert_eq!(rules[0].conditions.len(), 1);
    }
}

/// Axiom generated during universal quantifier removal.
#[derive(Debug, Clone)]
pub struct NormalizationAxiom {
    pub name: String,
    pub parameters: Vec<(String, Option<String>)>, // (name, type)
    pub condition: Condition,
}

/// Context for managing axioms and type maps during normalization.
#[derive(Debug, Clone)]
pub struct NormalizationContext {
    pub axioms: Vec<NormalizationAxiom>,
    axiom_counter: usize,
}

impl NormalizationContext {
    pub fn new() -> Self {
        Self {
            axioms: Vec::new(),
            axiom_counter: 0,
        }
    }

    /// Create a new axiom with the given parameters and condition.
    /// Returns the axiom name.
    pub fn add_axiom(
        &mut self,
        parameters: Vec<(String, Option<String>)>,
        condition: Condition,
    ) -> String {
        let name = format!("new-axiom@{}", self.axiom_counter);
        self.axiom_counter += 1;
        self.axioms.push(NormalizationAxiom {
            name: name.clone(),
            parameters,
            condition,
        });
        name
    }
}

/// Remove universal quantifiers from a condition.
///
/// This function implements the normalization step [1] from Python's normalize.py:
/// Replace, in a top-down fashion, <forall(vars, phi)> by <not(not-all-phi)>,
/// where <not-all-phi> is a new axiom.
///
/// The negation of forall(vars, phi) becomes exists(vars, not(phi)),
/// and we create an axiom for that existential condition.
///
/// Note: Python version includes caching of axioms by condition. For simplicity,
/// we skip caching here since Condition contains SExpr which doesn't implement Hash.
/// This may create duplicate axioms but maintains functional correctness.
pub fn remove_universal_quantifiers(
    condition: &Condition,
    type_map: &HashMap<String, String>,
    ctx: &mut NormalizationContext,
) -> Condition {
    match condition {
        Condition::Forall(_params, _inner) => {
            // Negate to get exists(params, not(inner))
            // Note: negate() recursively negates, so forall becomes exists,
            // and nested foralls also get negated
            let axiom_condition = condition.negate();

            // Collect free variables (parameters for the new axiom)
            let mut free_vars: Vec<String> = axiom_condition.free_variables().into_iter().collect();
            free_vars.sort();

            // Create typed parameters from free variables
            let typed_params: Vec<(String, Option<String>)> = free_vars
                .iter()
                .map(|v| {
                    let ty = type_map.get(v).cloned();
                    (v.clone(), ty)
                })
                .collect();

            // Recursively process the axiom condition to handle any nested foralls
            // that might appear after negation
            let processed_condition = remove_universal_quantifiers(&axiom_condition, type_map, ctx);

            // Add the axiom
            let axiom_name = ctx.add_axiom(typed_params, processed_condition);

            // Return a negated atom referencing the axiom
            Condition::Not(Box::new(Condition::Atom(axiom_name, free_vars)))
        }
        // For Exists, recursively process the inner condition
        Condition::Exists(params, inner) => {
            let processed_inner = remove_universal_quantifiers(inner, type_map, ctx);
            Condition::Exists(params.clone(), Box::new(processed_inner))
        }
        // For all other conditions, recursively process sub-parts
        _ => {
            let parts = condition.parts();
            if parts.is_empty() {
                condition.clone()
            } else {
                let new_parts: Vec<Condition> = parts
                    .iter()
                    .map(|p| remove_universal_quantifiers(p, type_map, ctx))
                    .collect();
                condition.change_parts(new_parts)
            }
        }
    }
}

#[cfg(test)]
mod quantifier_tests {
    use super::*;

    #[test]
    fn test_remove_simple_forall() {
        // forall(?x, at(?x, loc1)) becomes not(new-axiom@0())
        // where new-axiom@0 is exists(?x, not(at(?x, loc1)))
        // Since ?x is bound by the forall and has no free variables from outside,
        // the axiom has no parameters
        let mut ctx = NormalizationContext::new();
        let mut type_map = HashMap::new();
        type_map.insert("?x".to_string(), "object".to_string());

        let forall_cond = Condition::Forall(
            vec![("?x".to_string(), Some("object".to_string()))],
            Box::new(Condition::Atom("at".to_string(), vec!["?x".to_string(), "loc1".to_string()])),
        );

        let result = remove_universal_quantifiers(&forall_cond, &type_map, &mut ctx);

        // Should produce Not(Atom("new-axiom@0", []))
        assert!(matches!(result, Condition::Not(_)));
        if let Condition::Not(inner) = result {
            if let Condition::Atom(name, args) = &*inner {
                assert_eq!(name, "new-axiom@0");
                assert_eq!(args.len(), 0, "Axiom should have no parameters");
            } else {
                panic!("Expected Atom inside Not");
            }
        }

        // Check that one axiom was created
        assert_eq!(ctx.axioms.len(), 1);
        assert_eq!(ctx.axioms[0].name, "new-axiom@0");
        // The axiom's condition should be exists(?x, not(at(?x, loc1)))
        assert!(matches!(ctx.axioms[0].condition, Condition::Exists(_, _)));
    }

    #[test]
    fn test_nested_forall() {
        // forall(?x, forall(?y, connected(?x, ?y)))
        // When negated, this becomes: exists(?x, exists(?y, not(connected(?x, ?y))))
        // This creates one axiom with no parameters (both vars are bound)
        let mut ctx = NormalizationContext::new();
        let mut type_map = HashMap::new();
        type_map.insert("?x".to_string(), "loc".to_string());
        type_map.insert("?y".to_string(), "loc".to_string());

        let inner_forall = Condition::Forall(
            vec![("?y".to_string(), Some("loc".to_string()))],
            Box::new(Condition::Atom(
                "connected".to_string(),
                vec!["?x".to_string(), "?y".to_string()],
            )),
        );

        let outer_forall = Condition::Forall(
            vec![("?x".to_string(), Some("loc".to_string()))],
            Box::new(inner_forall),
        );

        let result = remove_universal_quantifiers(&outer_forall, &type_map, &mut ctx);

        // The nested foralls are negated together, creating one axiom
        assert!(matches!(result, Condition::Not(_)));
        assert_eq!(ctx.axioms.len(), 1);
        // The axiom should have nested Exists conditions
        assert!(matches!(ctx.axioms[0].condition, Condition::Exists(_, _)));
    }

    #[test]
    fn test_forall_with_free_variable() {
        // forall(?y, connected(?x, ?y)) where ?x is free
        // This should create an axiom with parameter ?x
        let mut ctx = NormalizationContext::new();
        let mut type_map = HashMap::new();
        type_map.insert("?x".to_string(), "loc".to_string());
        type_map.insert("?y".to_string(), "loc".to_string());

        let forall_cond = Condition::Forall(
            vec![("?y".to_string(), Some("loc".to_string()))],
            Box::new(Condition::Atom(
                "connected".to_string(),
                vec!["?x".to_string(), "?y".to_string()],
            )),
        );

        let result = remove_universal_quantifiers(&forall_cond, &type_map, &mut ctx);

        // Should produce Not(Atom("new-axiom@0", ["?x"]))
        // because ?x is a free variable in the forall
        assert!(matches!(result, Condition::Not(_)));
        if let Condition::Not(inner) = result {
            if let Condition::Atom(name, args) = &*inner {
                assert_eq!(name, "new-axiom@0");
                assert_eq!(args, &vec!["?x".to_string()]);
            } else {
                panic!("Expected Atom inside Not");
            }
        }

        // Check that one axiom was created with one parameter
        assert_eq!(ctx.axioms.len(), 1);
        assert_eq!(ctx.axioms[0].name, "new-axiom@0");
        assert_eq!(ctx.axioms[0].parameters.len(), 1);
        assert_eq!(ctx.axioms[0].parameters[0].0, "?x");
    }
}

/// Substitute complicated goal conditions with axioms.
///
/// This function implements the goal normalization from Python's normalize.py.
/// If the goal is:
/// - A simple literal (Atom or Not(Atom)): leave as-is
/// - A conjunction of only literals: leave as-is
/// - Otherwise: create an axiom for the goal and replace it with an atom referencing that axiom
///
/// This simplifies goal handling in the rest of the translation pipeline.
pub fn substitute_complicated_goal(
    goal: &Condition,
    ctx: &mut NormalizationContext,
) -> Condition {
    // Check if goal is a simple literal (Atom or Not(Atom))
    match goal {
        Condition::Atom(_, _) => return goal.clone(),
        Condition::Not(inner) => {
            if matches!(**inner, Condition::Atom(_, _)) {
                return goal.clone();
            }
        }
        Condition::And(parts) => {
            // Check if all parts are literals
            let all_literals = parts.iter().all(|part| match part {
                Condition::Atom(_, _) => true,
                Condition::Not(inner) => matches!(**inner, Condition::Atom(_, _)),
                _ => false,
            });
            if all_literals {
                return goal.clone();
            }
        }
        _ => {}
    }

    // Goal is complicated - create an axiom for it
    let axiom_name = ctx.add_axiom(vec![], goal.clone());
    Condition::Atom(axiom_name, vec![])
}

#[cfg(test)]
mod goal_tests {
    use super::*;

    #[test]
    fn test_simple_atom_goal_unchanged() {
        let mut ctx = NormalizationContext::new();
        let goal = Condition::Atom("at".to_string(), vec!["robot".to_string(), "loc1".to_string()]);
        let result = substitute_complicated_goal(&goal, &mut ctx);
        
        assert_eq!(result, goal);
        assert_eq!(ctx.axioms.len(), 0, "No axiom should be created");
    }

    #[test]
    fn test_negated_atom_goal_unchanged() {
        let mut ctx = NormalizationContext::new();
        let goal = Condition::Not(Box::new(Condition::Atom(
            "blocked".to_string(),
            vec!["door1".to_string()],
        )));
        let result = substitute_complicated_goal(&goal, &mut ctx);
        
        assert_eq!(result, goal);
        assert_eq!(ctx.axioms.len(), 0, "No axiom should be created");
    }

    #[test]
    fn test_conjunction_of_literals_unchanged() {
        let mut ctx = NormalizationContext::new();
        let goal = Condition::And(vec![
            Condition::Atom("at".to_string(), vec!["robot".to_string(), "loc1".to_string()]),
            Condition::Not(Box::new(Condition::Atom(
                "blocked".to_string(),
                vec!["door1".to_string()],
            ))),
        ]);
        let result = substitute_complicated_goal(&goal, &mut ctx);
        
        assert_eq!(result, goal);
        assert_eq!(ctx.axioms.len(), 0, "No axiom should be created");
    }

    #[test]
    fn test_disjunctive_goal_creates_axiom() {
        let mut ctx = NormalizationContext::new();
        let goal = Condition::Or(vec![
            Condition::Atom("at".to_string(), vec!["robot".to_string(), "loc1".to_string()]),
            Condition::Atom("at".to_string(), vec!["robot".to_string(), "loc2".to_string()]),
        ]);
        let result = substitute_complicated_goal(&goal, &mut ctx);
        
        // Should create an axiom and return an atom referencing it
        assert!(matches!(result, Condition::Atom(_, _)));
        if let Condition::Atom(name, args) = result {
            assert_eq!(name, "new-axiom@0");
            assert_eq!(args.len(), 0, "Goal axiom should have no parameters");
        }
        assert_eq!(ctx.axioms.len(), 1);
        assert_eq!(ctx.axioms[0].name, "new-axiom@0");
    }

    #[test]
    fn test_existential_goal_creates_axiom() {
        let mut ctx = NormalizationContext::new();
        let goal = Condition::Exists(
            vec![("?x".to_string(), Some("location".to_string()))],
            Box::new(Condition::Atom(
                "at".to_string(),
                vec!["robot".to_string(), "?x".to_string()],
            )),
        );
        let result = substitute_complicated_goal(&goal, &mut ctx);
        
        assert!(matches!(result, Condition::Atom(_, _)));
        assert_eq!(ctx.axioms.len(), 1);
    }

    #[test]
    fn test_conjunction_with_nested_condition_creates_axiom() {
        let mut ctx = NormalizationContext::new();
        let goal = Condition::And(vec![
            Condition::Atom("at".to_string(), vec!["robot".to_string(), "loc1".to_string()]),
            Condition::Or(vec![
                Condition::Atom("open".to_string(), vec!["door1".to_string()]),
                Condition::Atom("open".to_string(), vec!["door2".to_string()]),
            ]),
        ]);
        let result = substitute_complicated_goal(&goal, &mut ctx);
        
        // Contains a disjunction, so should create an axiom
        assert!(matches!(result, Condition::Atom(_, _)));
        assert_eq!(ctx.axioms.len(), 1);
    }
}

/// Convert condition to Disjunctive Normal Form (DNF).
///
/// This function implements the DNF transformation from Python's normalize.py.
/// After removing universal quantifiers, the following rules are applied:
/// (1) or(phi, or(psi, psi'))      ==  or(phi, psi, psi')         [Associativity]
/// (2) exists(vars, or(phi, psi))  ==  or(exists(vars, phi), exists(vars, psi)) [Distribution]
/// (3) and(phi, or(psi, psi'))     ==  or(and(phi, psi), and(phi, psi'))        [Distribution]
///
/// This pulls all disjunctions to the outermost level of the condition.
pub fn build_dnf(condition: &Condition) -> Condition {
    // Recursively process all parts first
    let processed_parts: Vec<Condition> = condition
        .parts()
        .iter()
        .map(|part| build_dnf(part))
        .collect();

    // If no parts, return as-is
    if processed_parts.is_empty() {
        return condition.clone();
    }

    // Separate disjunctive parts from other parts
    let mut disjunctive_parts: Vec<Condition> = Vec::new();
    let mut other_parts: Vec<Condition> = Vec::new();

    for part in processed_parts {
        if matches!(part, Condition::Or(_)) {
            disjunctive_parts.push(part);
        } else {
            other_parts.push(part);
        }
    }

    // If no disjunctive parts, just reconstruct with processed parts
    if disjunctive_parts.is_empty() {
        let all_parts: Vec<Condition> = condition
            .parts()
            .iter()
            .map(|part| build_dnf(part))
            .collect();
        return condition.change_parts(all_parts);
    }

    // Apply transformation rules based on the condition type
    match condition {
        // Rule (1): Associativity of disjunction
        // or(phi, or(psi, psi')) → or(phi, psi, psi')
        Condition::Or(_) => {
            let mut result_parts = other_parts;
            for part in disjunctive_parts {
                if let Condition::Or(inner_parts) = part {
                    result_parts.extend(inner_parts);
                }
            }
            Condition::Or(result_parts)
        }

        // Rule (2): Distributivity disjunction/existential quantification
        // exists(vars, or(phi, psi)) → or(exists(vars, phi), exists(vars, psi))
        Condition::Exists(params, _) => {
            if let Some(Condition::Or(or_parts)) = disjunctive_parts.first() {
                let result_parts: Vec<Condition> = or_parts
                    .iter()
                    .map(|part| Condition::Exists(params.clone(), Box::new(part.clone())))
                    .collect();
                Condition::Or(result_parts)
            } else {
                // Fallback - shouldn't happen
                condition.change_parts(
                    other_parts
                        .into_iter()
                        .chain(disjunctive_parts)
                        .collect(),
                )
            }
        }

        // Rule (3): Distributivity disjunction/conjunction
        // and(phi, or(psi, psi')) → or(and(phi, psi), and(phi, psi'))
        Condition::And(_) => {
            // Start with conjunction of non-disjunctive parts
            let mut result_parts = vec![Condition::And(other_parts.clone())];

            // Distribute each disjunctive part
            while let Some(disj) = disjunctive_parts.pop() {
                let previous_result_parts = result_parts;
                result_parts = Vec::new();

                if let Condition::Or(parts_to_distribute) = disj {
                    for part1 in &previous_result_parts {
                        for part2 in &parts_to_distribute {
                            // Create conjunction of part1 and part2
                            let new_conj = match part1 {
                                Condition::And(conj_parts) => {
                                    let mut new_parts = conj_parts.clone();
                                    new_parts.push(part2.clone());
                                    Condition::And(new_parts)
                                }
                                _ => Condition::And(vec![part1.clone(), part2.clone()]),
                            };
                            result_parts.push(new_conj);
                        }
                    }
                }
            }

            Condition::Or(result_parts)
        }

        // For other condition types, just return with processed parts
        _ => condition.change_parts(
            other_parts
                .into_iter()
                .chain(disjunctive_parts)
                .collect(),
        ),
    }
}

#[cfg(test)]
mod dnf_tests {
    use super::*;

    #[test]
    fn test_simple_condition_unchanged() {
        let cond = Condition::Atom("at".to_string(), vec!["robot".to_string(), "loc1".to_string()]);
        let result = build_dnf(&cond);
        assert_eq!(result, cond);
    }

    #[test]
    fn test_nested_disjunctions_flattened() {
        // or(A, or(B, C)) → or(A, B, C)
        let cond = Condition::Or(vec![
            Condition::Atom("a".to_string(), vec![]),
            Condition::Or(vec![
                Condition::Atom("b".to_string(), vec![]),
                Condition::Atom("c".to_string(), vec![]),
            ]),
        ]);
        let result = build_dnf(&cond);
        
        if let Condition::Or(parts) = result {
            assert_eq!(parts.len(), 3);
        } else {
            panic!("Expected Or condition");
        }
    }

    #[test]
    fn test_conjunction_with_disjunction_distributed() {
        // and(A, or(B, C)) → or(and(A, B), and(A, C))
        let cond = Condition::And(vec![
            Condition::Atom("a".to_string(), vec![]),
            Condition::Or(vec![
                Condition::Atom("b".to_string(), vec![]),
                Condition::Atom("c".to_string(), vec![]),
            ]),
        ]);
        let result = build_dnf(&cond);
        
        if let Condition::Or(parts) = result {
            assert_eq!(parts.len(), 2);
            // Each part should be a conjunction
            assert!(matches!(parts[0], Condition::And(_)));
            assert!(matches!(parts[1], Condition::And(_)));
        } else {
            panic!("Expected Or condition, got: {:?}", result);
        }
    }

    #[test]
    fn test_exists_with_disjunction_distributed() {
        // exists(?x, or(A(?x), B(?x))) → or(exists(?x, A(?x)), exists(?x, B(?x)))
        let cond = Condition::Exists(
            vec![("?x".to_string(), Some("obj".to_string()))],
            Box::new(Condition::Or(vec![
                Condition::Atom("a".to_string(), vec!["?x".to_string()]),
                Condition::Atom("b".to_string(), vec!["?x".to_string()]),
            ])),
        );
        let result = build_dnf(&cond);
        
        if let Condition::Or(parts) = result {
            assert_eq!(parts.len(), 2);
            assert!(matches!(parts[0], Condition::Exists(_, _)));
            assert!(matches!(parts[1], Condition::Exists(_, _)));
        } else {
            panic!("Expected Or condition");
        }
    }

    #[test]
    fn test_multiple_disjunctions_in_conjunction() {
        // and(or(A, B), or(C, D)) → or(and(A, C), and(A, D), and(B, C), and(B, D))
        let cond = Condition::And(vec![
            Condition::Or(vec![
                Condition::Atom("a".to_string(), vec![]),
                Condition::Atom("b".to_string(), vec![]),
            ]),
            Condition::Or(vec![
                Condition::Atom("c".to_string(), vec![]),
                Condition::Atom("d".to_string(), vec![]),
            ]),
        ]);
        let result = build_dnf(&cond);
        
        if let Condition::Or(parts) = result {
            // Should create all combinations: 2 * 2 = 4
            assert_eq!(parts.len(), 4);
        } else {
            panic!("Expected Or condition");
        }
    }
}
