//! Groups mutually exclusive atoms into finite-domain SAS variables.

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use tracing::info;

use super::invariant_finder;
use super::options;
use super::pddl::conditions::*;
use super::pddl::tasks::Task;
use super::symbols::ObjectId;

/// The facts of `group` that the delete relaxation can reach, with the counted
/// variable `?X` replaced by every object that makes the fact reachable.
///
/// A fact without a counted variable is subject to the same test: an invariant
/// found over the lifted task may name a fact that grounding proved
/// unreachable, and keeping it would put a fact into a SAS variable's domain
/// that no state ever holds.
type CountedFactIndex = HashMap<Atom, Vec<Atom>>;

fn index_counted_facts(task: &Task, reachable_facts: &HashSet<Atom>) -> CountedFactIndex {
    let object_order: HashMap<&str, usize> = task
        .objects
        .iter()
        .enumerate()
        .map(|(index, object)| (object.name.as_str(), index))
        .collect();
    let mut index: CountedFactIndex = HashMap::new();
    for fact in reachable_facts {
        for counted_position in 0..fact.args.len() {
            let mut counted_args = fact.args.clone();
            counted_args[counted_position] = "?X".to_string();
            index
                .entry(Atom::new(fact.predicate.clone(), counted_args))
                .or_default()
                .push(fact.clone());
        }
    }
    for (counted_fact, facts) in &mut index {
        let counted_position = counted_fact
            .args
            .iter()
            .position(|argument| argument == "?X")
            .expect("a counted-fact index key contains its counted argument");
        facts.sort_by_key(|fact| object_order[fact.args[counted_position].as_str()]);
    }
    index
}

fn expand_group(
    group: &[Atom],
    reachable_facts: &HashSet<Atom>,
    counted_facts: &CountedFactIndex,
) -> Vec<Atom> {
    let mut result = vec![];
    for fact in group {
        match fact.args.iter().position(|arg| arg == "?X") {
            Some(pos) => {
                debug_assert_eq!(fact.args[pos], "?X");
                if let Some(matches) = counted_facts.get(fact) {
                    result.extend(matches.iter().cloned());
                }
            }
            None => {
                if reachable_facts.contains(fact) {
                    result.push(fact.clone());
                }
            }
        }
    }
    result
}

fn instantiate_groups(
    groups: &[Vec<Atom>],
    task: &Task,
    reachable_facts: &HashSet<Atom>,
) -> Vec<Vec<Atom>> {
    let counted_facts = index_counted_facts(task, reachable_facts);
    groups
        .iter()
        .map(|g| expand_group(g, reachable_facts, &counted_facts))
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
    /// Builds the queue over `groups`, leaving `excluded` facts out of every
    /// one of them. A fact no group covers ends up in a binary variable of its
    /// own.
    fn new(groups: &[Vec<Atom>], excluded: &HashSet<Atom>) -> Self {
        let groups: Vec<HashSet<Atom>> = groups
            .iter()
            .map(|group| {
                group
                    .iter()
                    .filter(|fact| !excluded.contains(*fact))
                    .cloned()
                    .collect()
            })
            .collect();
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

fn choose_groups(
    groups: &[Vec<Atom>],
    reachable_facts: &HashSet<Atom>,
    negative_in_goal: &HashSet<Atom>,
) -> Vec<Vec<Atom>> {
    let mut queue = GroupCoverQueue::new(groups, negative_in_goal);
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

/// The finite-domain encoding of the reachable atoms.
pub struct FactGroups {
    /// One group per SAS variable: the facts it can take as values.
    pub groups: Vec<Vec<Atom>>,
    /// Every mutex group found, including those no variable was built from.
    pub mutex_groups: Vec<Vec<Atom>>,
    /// The human-readable name of each value of each variable.
    pub translation_key: Vec<Vec<String>>,
}

/// Builds the variables from the invariants the task admits.
///
/// An atom of `negative_in_goal` is kept out of every mutex group, which leaves
/// it to a binary variable of its own. Only there is its negation a single
/// variable/value pair; inside a larger group the negated goal would be a
/// disjunction over the group's other facts, which a SAS goal cannot express.
pub fn compute_groups(
    task: &Task,
    atoms: &HashSet<Atom>,
    reachable_action_params: &HashMap<String, Vec<Rc<[ObjectId]>>>,
    negative_in_goal: &HashSet<Atom>,
) -> FactGroups {
    let groups = invariant_finder::get_groups(task, reachable_action_params);

    info!("Instantiating groups...");
    let groups = instantiate_groups(&groups, task, atoms);

    let groups = sort_groups(groups);

    info!("Collecting mutex groups...");
    let mutex_groups = collect_all_mutex_groups(&groups, atoms);

    info!("Choosing groups...");
    let groups = choose_groups(&groups, atoms, negative_in_goal);

    let groups = sort_groups(groups);

    info!("Building translation key...");
    let translation_key = build_translation_key(&groups);

    FactGroups {
        groups,
        mutex_groups,
        translation_key,
    }
}

/// Builds one variable per fact, skipping invariant synthesis. The encoding is
/// larger but the task it describes is the same.
pub fn compute_singleton_groups(atoms: &HashSet<Atom>) -> FactGroups {
    let groups = sort_groups(atoms.iter().cloned().map(|atom| vec![atom]).collect());
    let translation_key = build_translation_key(&groups);

    FactGroups {
        mutex_groups: groups.clone(),
        groups,
        translation_key,
    }
}

#[cfg(test)]
mod tests {
    use super::{Atom, build_translation_key, choose_groups};
    use std::collections::HashSet;

    fn atom(predicate: &str, argument: &str) -> Atom {
        Atom::new(predicate.to_string(), vec![argument.to_string()])
    }

    /// No corpus task has a negative goal, so only a direct test pins the
    /// encoding such an atom needs.
    #[test]
    fn an_atom_negated_in_the_goal_gets_a_binary_variable_of_its_own() {
        let negated_in_goal = atom("tree", "cell6");
        let group = vec![
            negated_in_goal.clone(),
            atom("tree", "cell5"),
            atom("tree", "cell4"),
        ];
        let reachable: HashSet<Atom> = group.iter().cloned().collect();

        let selected = choose_groups(
            std::slice::from_ref(&group),
            &reachable,
            &HashSet::from([negated_in_goal.clone()]),
        );
        let key = build_translation_key(&selected);
        let covering: Vec<usize> = (0..selected.len())
            .filter(|&var| selected[var].contains(&negated_in_goal))
            .collect();

        // One variable covers the atom, and its only other value is the atom's
        // negation rather than one of the facts it is mutex with.
        assert_eq!(covering.len(), 1);
        assert_eq!(
            key[covering[0]],
            [
                format!("{negated_in_goal}"),
                format!("{}", negated_in_goal.negate())
            ]
        );
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

        let selected = choose_groups(&groups, &reachable, &HashSet::new());
        let occurrences = selected
            .iter()
            .flatten()
            .filter(|fact| **fact == shared)
            .count();

        assert_eq!(occurrences, 1);
    }
}
