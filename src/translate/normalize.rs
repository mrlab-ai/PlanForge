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
