/// Port of split_rules.py
/// Splits rules with many conditions into binary rules.

use std::collections::HashSet;
use super::pddl_to_prolog::{Rule, RuleType, get_variables};
use super::graph::Graph;

/// Python: def get_connected_conditions(conditions)
fn get_connected_conditions(conditions: &[Vec<String>]) -> Vec<Vec<usize>> {
    let n = conditions.len();
    let mut graph = Graph::new((0..n).collect());

    // Build var_to_conditions mapping
    for i in 0..n {
        for j in (i+1)..n {
            let vars_i: HashSet<&String> = conditions[i][1..].iter()
                .filter(|a| a.starts_with('?'))
                .collect();
            let vars_j: HashSet<&String> = conditions[j][1..].iter()
                .filter(|a| a.starts_with('?'))
                .collect();
            if vars_i.intersection(&vars_j).next().is_some() {
                graph.connect(i, j);
            }
        }
    }

    let components = graph.connected_components();
    let mut result: Vec<Vec<usize>> = components.into_iter()
        .map(|s| {
            let mut v: Vec<usize> = s.into_iter().collect();
            v.sort();
            v
        })
        .collect();
    result.sort();
    result
}

/// Python: def project_rule(rule, conditions, name_generator)
fn project_rule(
    rule: &Rule,
    condition_indices: &[usize],
    counter: &mut usize,
) -> Rule {
    let selected_conditions: Vec<Vec<String>> = condition_indices.iter()
        .map(|&i| rule.conditions[i].clone())
        .collect();

    let predicate = format!("p${}", counter);
    *counter += 1;

    let cond_vars = get_variables(&selected_conditions);
    let effect_vars: HashSet<String> = rule.effect[1..].iter()
        .filter(|a| a.starts_with('?'))
        .cloned()
        .collect();

    let mut result_vars: Vec<String> = cond_vars.intersection(&effect_vars)
        .cloned()
        .collect();
    result_vars.sort();

    let mut effect = vec![predicate];
    effect.extend(result_vars);

    Rule::new(selected_conditions, effect)
}

/// Python: def split_rule(rule, name_generator)
pub fn split_rule(rule: &Rule, counter: &mut usize) -> Vec<Rule> {
    // Separate important (have variables) from trivial (no variables) conditions
    let mut important_indices = vec![];
    let mut trivial_conditions = vec![];
    for (i, cond) in rule.conditions.iter().enumerate() {
        let has_var = cond[1..].iter().any(|a| a.starts_with('?'));
        if has_var {
            important_indices.push(i);
        } else {
            trivial_conditions.push(cond.clone());
        }
    }

    let important_conditions: Vec<Vec<String>> = important_indices.iter()
        .map(|&i| rule.conditions[i].clone())
        .collect();

    let components = get_connected_conditions(&important_conditions);
    if components.len() == 1 && trivial_conditions.is_empty() {
        return split_into_binary_rules(rule, counter);
    }

    // Map component indices back to original condition indices
    let components_original: Vec<Vec<usize>> = components.iter()
        .map(|comp| comp.iter().map(|&i| important_indices[i]).collect())
        .collect();

    let projected_rules: Vec<Rule> = components_original.iter()
        .map(|comp| project_rule(rule, comp, counter))
        .collect();

    let mut result = vec![];
    for proj_rule in &projected_rules {
        result.extend(split_into_binary_rules(proj_rule, counter));
    }

    let mut combining_conditions: Vec<Vec<String>> = projected_rules.iter()
        .map(|r| r.effect.clone())
        .collect();
    combining_conditions.extend(trivial_conditions);

    let mut combining_rule = Rule::new(combining_conditions.clone(), rule.effect.clone());
    if combining_conditions.len() >= 2 {
        combining_rule.rule_type = Some(RuleType::Product);
    } else {
        combining_rule.rule_type = Some(RuleType::Project);
    }
    result.push(combining_rule);

    result
}

/// Python: def split_into_binary_rules(rule, name_generator)
pub fn split_into_binary_rules(rule: &Rule, counter: &mut usize) -> Vec<Rule> {
    if rule.conditions.len() <= 1 {
        let mut r = rule.clone();
        r.rule_type = Some(RuleType::Project);
        return vec![r];
    }
    super::greedy_join::greedy_join(rule, counter)
}
