/// Port of fact_groups.py
/// Groups atoms into mutex groups / FDR variables.

use std::collections::{HashMap, HashSet};

use super::pddl::conditions::*;
use super::pddl::tasks::Task;
use super::invariant_finder;
use super::options;

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
fn instantiate_groups(groups: &[Vec<Atom>], task: &Task, reachable_facts: &HashSet<Atom>) -> Vec<Vec<Atom>> {
    groups.iter().map(|g| expand_group(g, task, reachable_facts)).collect()
}

/// Python: class GroupCoverQueue
struct GroupCoverQueue {
    max_size: usize,
    groups_by_size: Vec<Vec<HashSet<Atom>>>,
    groups_by_fact: HashMap<Atom, Vec<usize>>, // indices into a flat list
    top: Option<HashSet<Atom>>,
    all_groups: Vec<HashSet<Atom>>,
}

impl GroupCoverQueue {
    fn new(groups: &[Vec<Atom>]) -> Self {
        if groups.is_empty() {
            return GroupCoverQueue {
                max_size: 0,
                groups_by_size: vec![],
                groups_by_fact: HashMap::new(),
                top: None,
                all_groups: vec![],
            };
        }

        let max_size = groups.iter().map(|g| g.len()).max().unwrap_or(0);
        let mut groups_by_size: Vec<Vec<HashSet<Atom>>> = vec![vec![]; max_size + 1];
        let mut groups_by_fact: HashMap<Atom, Vec<usize>> = HashMap::new();
        let mut all_groups: Vec<HashSet<Atom>> = vec![];

        for group in groups {
            let group_set: HashSet<Atom> = group.iter().cloned().collect();
            let idx = all_groups.len();
            groups_by_size[group_set.len()].push(group_set.clone());
            for fact in &group_set {
                groups_by_fact.entry(fact.clone()).or_default().push(idx);
            }
            all_groups.push(group_set);
        }

        let mut q = GroupCoverQueue {
            max_size,
            groups_by_size,
            groups_by_fact,
            top: None,
            all_groups,
        };
        q.update_top();
        q
    }

    fn is_active(&self) -> bool {
        self.max_size > 1
    }

    fn pop(&mut self) -> Vec<Atom> {
        let result: Vec<Atom> = self.top.take().unwrap().into_iter().collect();
        if options::USE_PARTIAL_ENCODING {
            for fact in &result {
                // Remove fact from all groups that contain it
                // This is a simplified version - in Python it modifies groups in-place
                for group in &mut self.all_groups {
                    group.remove(fact);
                }
            }
        }
        self.update_top();
        result
    }

    fn update_top(&mut self) {
        while self.max_size > 1 {
            // Collect candidates to redistribute
            let mut to_redistribute: Vec<HashSet<Atom>> = vec![];
            let mut found: Option<HashSet<Atom>> = None;

            while let Some(candidate) = self.groups_by_size[self.max_size].pop() {
                if candidate.len() == self.max_size {
                    found = Some(candidate);
                    break;
                }
                if !candidate.is_empty() {
                    to_redistribute.push(candidate);
                }
            }

            for cand in to_redistribute {
                let sz = cand.len();
                self.groups_by_size[sz].push(cand);
            }

            if found.is_some() {
                self.top = found;
                return;
            }
            self.max_size -= 1;
        }
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
    println!("{} uncovered facts", uncovered_facts.len());
    for fact in &uncovered_facts {
        result.push(vec![fact.clone()]);
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

/// Python: def sort_groups(groups)
fn sort_groups(groups: Vec<Vec<Atom>>) -> Vec<Vec<Atom>> {
    let mut sorted: Vec<Vec<Atom>> = groups.into_iter()
        .map(|mut g| {
            g.sort_by(|a, b| format!("{:?}", a).cmp(&format!("{:?}", b)));
            g
        })
        .collect();
    sorted.sort_by(|a, b| format!("{:?}", a).cmp(&format!("{:?}", b)));
    sorted
}

/// Python: def compute_groups(task, atoms, reachable_action_params)
/// Returns (groups, mutex_groups, translation_key)
pub fn compute_groups(
    task: &Task,
    atoms: &HashSet<Atom>,
    reachable_action_params: &Option<HashMap<String, Vec<Vec<String>>>>,
) -> (Vec<Vec<Atom>>, Vec<Vec<Atom>>, Vec<Vec<String>>) {
    let groups = invariant_finder::get_groups(task, reachable_action_params);

    println!("Instantiating groups...");
    let groups = instantiate_groups(&groups, task, atoms);

    let groups = sort_groups(groups);

    println!("Collecting mutex groups...");
    let mutex_groups = collect_all_mutex_groups(&groups, atoms);

    println!("Choosing groups...");
    let groups = choose_groups(&groups, atoms);

    let groups = sort_groups(groups);

    println!("Building translation key...");
    let translation_key = build_translation_key(&groups);

    (groups, mutex_groups, translation_key)
}
