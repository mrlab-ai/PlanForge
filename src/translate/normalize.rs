use crate::translate::build_model as bm;
use std::collections::HashSet;

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
					args: rule.effect
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
			conditions: conds.into_iter().map(|(p, args)| sym_atom(p, args)).collect(),
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
