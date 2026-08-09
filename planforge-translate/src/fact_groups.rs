//! Groups mutually exclusive atoms into finite-domain SAS variables.

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

use tracing::info;

use super::invariant_finder;
use super::options;
use super::pddl::conditions::*;
use super::pddl::tasks::Task;

/// Python: def expand_group(group, task, reachable_facts)
fn expand_group(group: &[Atom], task: &Task, reachable_facts: &HashSet<Atom>) -> Vec<Atom> {
    let mut result = vec![];
    for fact in group {
        if let Some(pos) = fact.args.iter().position(|a| a == "?X") {
            for obj in &task.objects {
                let mut newargs = fact.args.clone();
                newargs[pos] = obj.name.clone();
                let atom = Atom::new(fact.predicate.clone(), newargs);
                if reachable_facts.contains(&atom) {
                    result.push(atom);
                }
            }
        } else {
            result.push(fact.clone());
        }
    }
    result
}

/// Python: def instantiate_groups(groups, task, reachable_facts)
fn instantiate_groups(
    groups: &[Vec<Atom>],
    task: &Task,
    reachable_facts: &HashSet<Atom>,
) -> Vec<Vec<Atom>> {
    groups
        .iter()
        .map(|g| expand_group(g, task, reachable_facts))
        .collect()
}

/// Greedily hands out the largest remaining mutex group.
///
/// Under partial encoding a fact may be covered by only one selected group, so
/// selecting a group shrinks every other group sharing a fact with it. Groups
/// therefore live in one arena and are referred to by index: the size buckets
/// hold indices, and `groups_by_fact` says which groups a popped fact has to
/// be removed from, instead of every group being scanned on every pop.
struct GroupCoverQueue {
    groups: Vec<HashSet<Atom>>,
    groups_by_fact: HashMap<Atom, Vec<usize>>,
    max_size: usize,
    by_size: Vec<Vec<usize>>,
    top: Option<usize>,
}

impl GroupCoverQueue {
    fn new(groups: &[Vec<Atom>]) -> Self {
        let groups: Vec<HashSet<Atom>> =
            groups.iter().map(|g| g.iter().cloned().collect()).collect();
        let max_size = groups.iter().map(HashSet::len).max().unwrap_or(0);
        let mut by_size: Vec<Vec<usize>> = vec![vec![]; max_size + 1];
        let mut groups_by_fact: HashMap<Atom, Vec<usize>> = HashMap::new();
        for (index, group) in groups.iter().enumerate() {
            by_size[group.len()].push(index);
            for fact in group {
                groups_by_fact.entry(fact.clone()).or_default().push(index);
            }
        }

        let mut queue = GroupCoverQueue {
            groups,
            groups_by_fact,
            max_size,
            by_size,
            top: None,
        };
        queue.update_top();
        queue
    }

    fn is_active(&self) -> bool {
        self.max_size > 1
    }

    fn pop(&mut self) -> Vec<Atom> {
        let selected = std::mem::take(&mut self.groups[self.top.take().expect("queue is active")]);
        if options::USE_PARTIAL_ENCODING {
            for fact in &selected {
                for &index in &self.groups_by_fact[fact] {
                    self.groups[index].remove(fact);
                }
            }
        }
        self.update_top();
        selected.into_iter().collect()
    }

    /// Finds the largest group still holding as many facts as its bucket
    /// claims, moving the ones that have shrunk to their real bucket.
    fn update_top(&mut self) {
        while self.max_size > 1 {
            let mut shrunk: Vec<(usize, usize)> = vec![];
            let mut found = None;
            while let Some(candidate) = self.by_size[self.max_size].pop() {
                let size = self.groups[candidate].len();
                if size == self.max_size {
                    found = Some(candidate);
                    break;
                }
                if size > 0 {
                    shrunk.push((size, candidate));
                }
            }
            for (size, candidate) in shrunk {
                self.by_size[size].push(candidate);
            }
            if found.is_some() {
                self.top = found;
                return;
            }
            self.max_size -= 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Atom, choose_groups};
    use std::collections::HashSet;

    fn atom(predicate: &str, argument: &str) -> Atom {
        Atom::new(predicate.to_string(), vec![argument.to_string()])
    }

    #[test]
    fn partial_encoding_removes_selected_facts_from_queued_groups() {
        let shared = atom("tree", "cell6");
        let left = atom("tree", "cell5");
        let right = atom("crafting_table", "cell6");
        let groups = vec![
            vec![shared.clone(), left.clone()],
            vec![shared.clone(), right.clone()],
        ];
        let reachable = HashSet::from([shared.clone(), left, right]);

        let selected = choose_groups(&groups, &reachable);
        let occurrences = selected
            .iter()
            .flatten()
            .filter(|fact| **fact == shared)
            .count();

        assert_eq!(occurrences, 1);
    }
}

/// Python: def choose_groups(groups, reachable_facts)
fn choose_groups(groups: &[Vec<Atom>], reachable_facts: &HashSet<Atom>) -> Vec<Vec<Atom>> {
    let mut queue = GroupCoverQueue::new(groups);
    let mut uncovered_facts = reachable_facts.clone();
    let mut result = vec![];
    while queue.is_active() {
        let group = queue.pop();
        for fact in &group {
            uncovered_facts.remove(fact);
        }
        result.push(group);
    }
    info!("{} uncovered facts", uncovered_facts.len());
    for fact in &uncovered_facts {
        result.push(vec![fact.clone()]);
    }
    if options::USE_PARTIAL_ENCODING {
        let mut seen = HashSet::new();
        for group in &result {
            for fact in group {
                assert!(
                    seen.insert(fact),
                    "partial encoding selected overlapping groups for {fact:?}: {result:?}"
                );
            }
        }
    }
    result
}

/// Python: def build_translation_key(groups)
pub fn build_translation_key(groups: &[Vec<Atom>]) -> Vec<Vec<String>> {
    let mut translation_keys = vec![];
    for group in groups {
        let mut group_key: Vec<String> = group.iter().map(|f| format!("{}", f)).collect();
        if group.len() == 1 {
            group_key.push(format!("{}", group[0].negate()));
        } else {
            group_key.push("<none of those>".to_string());
        }
        translation_keys.push(group_key);
    }
    translation_keys
}

/// Python: def collect_all_mutex_groups(groups, atoms)
fn collect_all_mutex_groups(groups: &[Vec<Atom>], atoms: &HashSet<Atom>) -> Vec<Vec<Atom>> {
    let mut all_groups = vec![];
    let mut uncovered_facts = atoms.clone();
    for group in groups {
        for fact in group {
            uncovered_facts.remove(fact);
        }
        all_groups.push(group.clone());
    }
    for fact in &uncovered_facts {
        all_groups.push(vec![fact.clone()]);
    }
    all_groups
}

/// Orders groups, and the facts inside each group, so that the SAS variable
/// order does not depend on the hash seed.
fn sort_groups(mut groups: Vec<Vec<Atom>>) -> Vec<Vec<Atom>> {
    for group in &mut groups {
        group.sort_unstable_by(cmp_atoms);
    }
    groups.sort_unstable_by(|left, right| {
        left.iter()
            .zip(right)
            .find_map(|(left, right)| match cmp_atoms(left, right) {
                Ordering::Equal => None,
                order => Some(order),
            })
            .unwrap_or_else(|| right.len().cmp(&left.len()))
    });
    groups
}

/// Python: def compute_groups(task, atoms, reachable_action_params)
/// Returns (groups, mutex_groups, translation_key)
pub fn compute_groups(
    task: &Task,
    atoms: &HashSet<Atom>,
    reachable_action_params: &Option<HashMap<String, Vec<Vec<String>>>>,
) -> (Vec<Vec<Atom>>, Vec<Vec<Atom>>, Vec<Vec<String>>) {
    let groups = invariant_finder::get_groups(task, reachable_action_params);

    info!("Instantiating groups...");
    let groups = instantiate_groups(&groups, task, atoms);

    let groups = sort_groups(groups);

    info!("Collecting mutex groups...");
    let mutex_groups = collect_all_mutex_groups(&groups, atoms);

    info!("Choosing groups...");
    let groups = choose_groups(&groups, atoms);

    let groups = sort_groups(groups);

    info!("Building translation key...");
    let translation_key = build_translation_key(&groups);

    (groups, mutex_groups, translation_key)
}

pub fn compute_singleton_groups(
    atoms: &HashSet<Atom>,
) -> (Vec<Vec<Atom>>, Vec<Vec<Atom>>, Vec<Vec<String>>) {
    let mut groups: Vec<Vec<Atom>> = atoms.iter().cloned().map(|atom| vec![atom]).collect();
    groups = sort_groups(groups);

    let mutex_groups = groups.clone();
    let translation_key = build_translation_key(&groups);

    (groups, mutex_groups, translation_key)
}
