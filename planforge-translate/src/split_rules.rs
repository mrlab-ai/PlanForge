//! Splits a rule with many conditions into rules with at most two.

use std::collections::HashMap;
use std::collections::HashSet;

use super::pddl_to_prolog::{Rule, RuleType, get_variables};

/// Representative of `node`'s component, halving the path on the way up.
fn find(parent: &mut [usize], mut node: usize) -> usize {
    while parent[node] != node {
        parent[node] = parent[parent[node]];
        node = parent[node];
    }
    node
}

/// Partitions the conditions into maximal sets connected by a shared variable.
///
/// Components come out sorted, and sorted by their smallest condition, because
/// the rules built from them end up in the program in this order. Union-find
/// over the variables replaces comparing every pair of conditions, which built
/// two hash sets per pair.
fn get_connected_conditions(conditions: &[Vec<String>]) -> Vec<Vec<usize>> {
    let mut parent: Vec<usize> = (0..conditions.len()).collect();
    let mut first_use: HashMap<&str, usize> = HashMap::new();
    for (index, condition) in conditions.iter().enumerate() {
        for variable in condition[1..].iter().filter(|arg| arg.starts_with('?')) {
            let earlier = *first_use.entry(variable).or_insert(index);
            let (left, right) = (find(&mut parent, earlier), find(&mut parent, index));
            // The smaller index wins, so a component's representative is its
            // smallest condition and the components below come out ordered.
            parent[left.max(right)] = left.min(right);
        }
    }

    let mut components: Vec<Vec<usize>> = vec![Vec::new(); conditions.len()];
    for index in 0..conditions.len() {
        let root = find(&mut parent, index);
        components[root].push(index);
    }
    components.retain(|component| !component.is_empty());
    components
}

/// Python: def project_rule(rule, conditions, name_generator)
fn project_rule(rule: &Rule, condition_indices: &[usize], counter: &mut usize) -> Rule {
    let selected_conditions: Vec<Vec<String>> = condition_indices
        .iter()
        .map(|&i| rule.conditions[i].clone())
        .collect();

    let predicate = format!("p${}", counter);
    *counter += 1;

    let cond_vars = get_variables(&selected_conditions);
    let effect_vars: HashSet<String> = rule.effect[1..]
        .iter()
        .filter(|a| a.starts_with('?'))
        .cloned()
        .collect();

    let mut result_vars: Vec<String> = cond_vars.intersection(&effect_vars).cloned().collect();
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

    let important_conditions: Vec<Vec<String>> = important_indices
        .iter()
        .map(|&i| rule.conditions[i].clone())
        .collect();

    let components = get_connected_conditions(&important_conditions);
    if components.len() == 1 && trivial_conditions.is_empty() {
        return split_into_binary_rules(rule, counter);
    }

    // Map component indices back to original condition indices
    let components_original: Vec<Vec<usize>> = components
        .iter()
        .map(|comp| comp.iter().map(|&i| important_indices[i]).collect())
        .collect();

    let projected_rules: Vec<Rule> = components_original
        .iter()
        .map(|comp| project_rule(rule, comp, counter))
        .collect();

    let mut result = vec![];
    for proj_rule in &projected_rules {
        result.extend(split_into_binary_rules(proj_rule, counter));
    }

    let mut combining_conditions: Vec<Vec<String>> =
        projected_rules.iter().map(|r| r.effect.clone()).collect();
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
