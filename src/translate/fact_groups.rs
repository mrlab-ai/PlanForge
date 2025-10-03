use crate::translate::pddl_parser::SExpr;
use crate::translate::pddl_ast::Condition;
use std::collections::{HashMap, HashSet};

/// Simplified fact grouping: group grounded atoms by predicate and first argument
/// for common binary predicates like at(item, place) -> group all at(item, *)
/// Falls back to singleton groups for anything else.
pub fn compute_groups_from_atoms(atoms: &Vec<String>) -> (Vec<Vec<String>>, Vec<Vec<String>>, Vec<Vec<String>>) {
    // groups, mutex_groups (same as groups here), translation_key (list of value names per group)
    let mut by_key: HashMap<String, Vec<String>> = HashMap::new();
    let mut remaining: HashSet<String> = atoms.iter().cloned().collect();

    for atom in atoms {
        // parse like "pred(arg1, arg2, ...)"
        if let Some(open) = atom.find('(') {
            if let Some(close) = atom.rfind(')') {
                let pred = &atom[..open];
                let args = &atom[open+1..close];
                let parts: Vec<&str> = args.split(',').map(|s| s.trim()).collect();
                if parts.len() >= 2 {
                    let key = format!("{}({})", pred, parts[0]);
                    by_key.entry(key).or_default().push(atom.clone());
                    remaining.remove(atom);
                    continue;
                }
            }
        }
        // fallback: singleton grouping by atom
        by_key.entry(atom.clone()).or_default().push(atom.clone());
        remaining.remove(atom);
    }

    // build groups list, deduplicate and sort each group for determinism
    let mut groups: Vec<Vec<String>> = by_key.into_iter().map(|(_k,v)| {
        let mut set: std::collections::HashSet<String> = v.into_iter().collect();
        let mut vec: Vec<String> = set.drain().collect();
        vec.sort();
        vec
    }).collect();
    groups.sort_by(|a,b| a.len().cmp(&b.len()).reverse());
    // mutex_groups: for now same as groups
    let mutex_groups = groups.clone();

    // translation_key: for each group, return the positive atom strings only.
    let translation_key: Vec<Vec<String>> = groups.clone();

    (groups, mutex_groups, translation_key)
}
