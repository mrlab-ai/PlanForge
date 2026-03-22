use crate::translate::build_model;
use crate::translate::greedy_join::greedy_join;

#[derive(Debug, Clone)]
pub struct SymRule {
    pub conditions: Vec<build_model::SymAtom>,
    pub effect: build_model::SymAtom,
}

#[derive(Debug, Clone)]
pub struct RuleWithType {
    pub rtype: String,
    pub conditions: Vec<build_model::SymAtom>,
    pub effect: build_model::SymAtom,
}

fn symatom_key(atom: &build_model::SymAtom) -> String {
    let mut key = atom.predicate.clone();
    key.push('(');
    key.push_str(&atom.args.join(","));
    key.push(')');
    key
}

fn get_variables(atom: &build_model::SymAtom) -> std::collections::HashSet<String> {
    atom.args
        .iter()
        .filter(|a| a.starts_with('?'))
        .cloned()
        .collect()
}

fn get_variables_in_atoms(atoms: &[build_model::SymAtom]) -> std::collections::HashSet<String> {
    let mut result = std::collections::HashSet::new();
    for atom in atoms {
        result.extend(get_variables(atom));
    }
    result
}

pub fn split_duplicate_arguments(rule: &mut SymRule) {
    fn rename_in_atom(
        atom: &build_model::SymAtom,
        extra_conditions: &mut Vec<build_model::SymAtom>,
    ) -> build_model::SymAtom {
        let mut used: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut new_args = atom.args.clone();
        for (i, arg) in atom.args.iter().enumerate() {
            if arg.starts_with('?') {
                if used.contains(arg) {
                    let new_var_name = format!("{}@{}", arg, extra_conditions.len());
                    new_args[i] = new_var_name.clone();
                    extra_conditions.push(build_model::SymAtom::new(
                        "=".to_string(),
                        vec![arg.clone(), new_var_name],
                    ));
                } else {
                    used.insert(arg.clone());
                }
            }
        }
        build_model::SymAtom::new(atom.predicate.clone(), new_args)
    }

    let mut extra_conditions: Vec<build_model::SymAtom> = Vec::new();
    rule.effect = rename_in_atom(&rule.effect, &mut extra_conditions);
    let mut new_conditions = Vec::new();
    for cond in &rule.conditions {
        new_conditions.push(rename_in_atom(cond, &mut extra_conditions));
    }
    new_conditions.extend(extra_conditions);
    rule.conditions = new_conditions;
}

pub fn add_object_conditions_to_rules(rules: &mut [SymRule]) -> bool {
    let mut inserted = false;
    for rule in rules.iter_mut() {
        let mut bound_vars: std::collections::HashSet<String> = std::collections::HashSet::new();
        for cond in &rule.conditions {
            for arg in &cond.args {
                if arg.starts_with('?') {
                    bound_vars.insert(arg.clone());
                }
            }
        }
        let mut extra_conditions: Vec<build_model::SymAtom> = Vec::new();
        for arg in &rule.effect.args {
            if !arg.starts_with('?') || bound_vars.contains(arg) {
                continue;
            }
            extra_conditions.push(build_model::SymAtom::new(
                "@object".to_string(),
                vec![arg.clone()],
            ));
            bound_vars.insert(arg.clone());
        }
        if !extra_conditions.is_empty() {
            rule.conditions.extend(extra_conditions);
            inserted = true;
        }
    }
    inserted
}

pub fn convert_trivial_rules_to_facts(rules: &[SymRule]) -> (Vec<SymRule>, Vec<build_model::Atom>) {
    let mut new_rules = Vec::new();
    let mut extra_facts = Vec::new();
    for rule in rules {
        if rule.conditions.is_empty() {
            let has_vars = rule.effect.args.iter().any(|a| a.starts_with('?'));
            if !has_vars {
                extra_facts.push(build_model::Atom {
                    predicate: rule.effect.predicate.clone(),
                    args: rule
                        .effect
                        .args
                        .iter()
                        .map(|s| build_model::Arg::Const(s.clone()))
                        .collect(),
                });
                continue;
            }
        }
        new_rules.push(rule.clone());
    }
    (new_rules, extra_facts)
}

fn get_connected_conditions(conditions: &[build_model::SymAtom]) -> Vec<Vec<build_model::SymAtom>> {
    let mut var_to_conditions: std::collections::HashMap<String, Vec<usize>> =
        std::collections::HashMap::new();
    for (idx, cond) in conditions.iter().enumerate() {
        for var in cond.args.iter().filter(|a| a.starts_with('?')) {
            var_to_conditions.entry(var.clone()).or_default().push(idx);
        }
    }

    let mut adjacency: Vec<std::collections::HashSet<usize>> =
        vec![std::collections::HashSet::new(); conditions.len()];
    for indices in var_to_conditions.values() {
        if indices.len() < 2 {
            continue;
        }
        let first = indices[0];
        for &other in &indices[1..] {
            adjacency[first].insert(other);
            adjacency[other].insert(first);
        }
    }

    let mut visited = vec![false; conditions.len()];
    let mut components: Vec<Vec<build_model::SymAtom>> = Vec::new();
    for i in 0..conditions.len() {
        if visited[i] {
            continue;
        }
        let mut stack = vec![i];
        let mut comp_indices = Vec::new();
        visited[i] = true;
        while let Some(node) = stack.pop() {
            comp_indices.push(node);
            for &nbr in &adjacency[node] {
                if !visited[nbr] {
                    visited[nbr] = true;
                    stack.push(nbr);
                }
            }
        }
        comp_indices.sort();
        let mut comp: Vec<build_model::SymAtom> = comp_indices
            .iter()
            .map(|&idx| conditions[idx].clone())
            .collect();
        comp.sort_by_key(symatom_key);
        components.push(comp);
    }
    components.sort_by(|a, b| symatom_key(&a[0]).cmp(&symatom_key(&b[0])));
    components
}

fn project_rule(
    rule: &SymRule,
    conditions: Vec<build_model::SymAtom>,
    counter: &mut usize,
) -> SymRule {
    let mut effect_vars: Vec<String> = rule
        .effect
        .args
        .iter()
        .filter(|a| a.starts_with('?'))
        .cloned()
        .collect();
    let cond_vars = get_variables_in_atoms(&conditions);
    effect_vars.retain(|v| cond_vars.contains(v));
    effect_vars.sort();
    let predicate = format!("p${}", counter);
    *counter += 1;
    let effect = build_model::SymAtom::new(predicate, effect_vars);
    SymRule { conditions, effect }
}

fn split_into_binary_rules(rule: &SymRule, counter: &mut usize) -> Vec<RuleWithType> {
    if rule.conditions.len() <= 1 {
        return vec![RuleWithType {
            rtype: "project".to_string(),
            conditions: rule.conditions.clone(),
            effect: rule.effect.clone(),
        }];
    }
    greedy_join(rule, counter)
}

pub fn split_rule(rule: &SymRule, counter: &mut usize) -> Vec<RuleWithType> {
    let mut important_conditions = Vec::new();
    let mut trivial_conditions = Vec::new();
    for cond in &rule.conditions {
        if cond.args.iter().any(|a| a.starts_with('?')) {
            important_conditions.push(cond.clone());
        } else {
            trivial_conditions.push(cond.clone());
        }
    }

    let components = get_connected_conditions(&important_conditions);
    if components.len() == 1 && trivial_conditions.is_empty() {
        return split_into_binary_rules(rule, counter);
    }

    let mut result = Vec::new();
    let mut projected_rules = Vec::new();
    for conditions in components {
        projected_rules.push(project_rule(rule, conditions, counter));
    }

    for proj in &projected_rules {
        result.extend(split_into_binary_rules(proj, counter));
    }

    let mut combined_conditions: Vec<build_model::SymAtom> =
        projected_rules.iter().map(|r| r.effect.clone()).collect();
    combined_conditions.extend(trivial_conditions);
    let rtype = if combined_conditions.len() >= 2 {
        "product".to_string()
    } else {
        "project".to_string()
    };
    result.push(RuleWithType {
        rtype,
        conditions: combined_conditions,
        effect: rule.effect.clone(),
    });
    result
}
