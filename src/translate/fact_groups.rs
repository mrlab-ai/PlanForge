use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use std::cell::RefCell;

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

// Expand a group with ?X into concrete atoms present in reachable_facts
fn expand_group(group: &Vec<String>, _domain: &crate::translate::pddl::Domain, problem: &crate::translate::pddl::Problem, reachable_facts: &HashSet<String>) -> Vec<String> {
    let mut result: Vec<String> = Vec::new();
    for fact in group {
        if fact.contains("?X") {
            // replace first ?X with each object name and keep if present in reachable_facts
            for (obj_name, _tp) in &problem.objects {
                let concrete = fact.replacen("?X", obj_name, 1);
                if reachable_facts.contains(&concrete) {
                    result.push(concrete);
                }
            }
        } else if reachable_facts.contains(fact) {
            result.push(fact.clone());
        }
    }
    result
}

pub fn instantiate_groups(groups: &Vec<Vec<String>>, domain: &crate::translate::pddl::Domain, problem: &crate::translate::pddl::Problem, reachable_facts: &HashSet<String>) -> Vec<Vec<String>> {
    groups.iter().map(|g| expand_group(g, domain, problem, reachable_facts)).collect()
}

pub struct GroupCoverQueue {
    groups_by_size: Vec<Vec<Rc<RefCell<HashSet<String>>>>>,
    groups_by_fact: HashMap<String, Vec<Rc<RefCell<HashSet<String>>>>>,
    max_size: usize,
    top: Option<Rc<RefCell<HashSet<String>>>>,
}

impl GroupCoverQueue {
    pub fn new(groups: Vec<HashSet<String>>) -> Self {
        if groups.is_empty() {
            return GroupCoverQueue { groups_by_size: Vec::new(), groups_by_fact: HashMap::new(), max_size: 0, top: None };
        }
        let max_size = groups.iter().map(|g| g.len()).max().unwrap_or(0);
        let mut groups_by_size: Vec<Vec<Rc<RefCell<HashSet<String>>>>> = vec![Vec::new(); max_size + 1];
        let mut groups_by_fact: HashMap<String, Vec<Rc<RefCell<HashSet<String>>>>> = HashMap::new();
        for g in groups.into_iter() {
            let sz = g.len();
            let rc = Rc::new(RefCell::new(g));
            for fact in rc.borrow().iter() {
                groups_by_fact.entry(fact.clone()).or_default().push(rc.clone());
            }
            groups_by_size[sz].push(rc);
        }
        let mut qc = GroupCoverQueue { groups_by_size, groups_by_fact, max_size, top: None };
        qc.update_top();
        qc
    }

    fn update_top(&mut self) {
        while self.max_size > 1 {
            // take from the size-bucket and check actual size; if changed, move it
            while let Some(candidate_rc) = self.groups_by_size[self.max_size].pop() {
                let cur_len = candidate_rc.borrow().len();
                if cur_len == self.max_size {
                    self.top = Some(candidate_rc.clone());
                    return;
                } else {
                    // move to bucket matching its new size
                    if cur_len > 0 {
                        self.groups_by_size[cur_len].push(candidate_rc.clone());
                    }
                }
            }
            self.max_size -= 1;
        }
        self.top = None;
    }

    pub fn pop(&mut self, use_partial_encoding: bool) -> Option<Vec<String>> {
        if let Some(top_rc) = self.top.take() {
            let result: Vec<String> = top_rc.borrow().iter().cloned().collect();
            if use_partial_encoding {
                // remove each fact from its groups
                for fact in &result {
                    if let Some(list) = self.groups_by_fact.get(fact) {
                        for g in list.iter() {
                            g.borrow_mut().remove(fact);
                        }
                    }
                }
            }
            self.update_top();
            return Some(result);
        }
        None
    }

    pub fn is_empty(&self) -> bool { self.top.is_none() }
}

pub fn choose_groups(groups: Vec<Vec<String>>, _domain: &crate::translate::pddl::Domain, _problem: &crate::translate::pddl::Problem, reachable_facts: &HashSet<String>, use_partial_encoding: bool) -> Vec<Vec<String>> {
    // convert groups to HashSet forms
    let mutable_groups: Vec<HashSet<String>> = groups.into_iter().map(|g| g.into_iter().collect()).collect();
    let mut queue = GroupCoverQueue::new(mutable_groups);
    let mut uncovered: HashSet<String> = reachable_facts.clone();
    let mut result: Vec<Vec<String>> = Vec::new();
    while !queue.is_empty() {
        if let Some(group) = queue.pop(use_partial_encoding) {
            for f in &group { uncovered.remove(f); }
            result.push(group);
        } else { break; }
    }
    // leftover uncovered facts become singleton groups
    for fact in uncovered {
        result.push(vec![fact]);
    }
    result
}

pub fn build_translation_key(groups: &Vec<Vec<String>>) -> Vec<Vec<String>> {
    groups.iter().map(|group| {
        // For multi-valued groups we return exactly the positive atom strings
        // (matching the reference SAS output). For singleton groups, return
        // the positive atom followed by its NegatedAtom counterpart.
        if group.len() == 1 {
            vec![group[0].clone(), format!("NegatedAtom {}", group[0])]
        } else {
            let key = group.clone();
            key
        }
    }).collect()
}

pub fn collect_all_mutex_groups(groups: &Vec<Vec<String>>, atoms: &HashSet<String>) -> Vec<Vec<String>> {
    let mut all_groups: Vec<Vec<String>> = Vec::new();
    let mut uncovered: HashSet<String> = atoms.clone();
    for group in groups {
        let gset: HashSet<String> = group.iter().cloned().collect();
        for fact in &gset { uncovered.remove(fact); }
        all_groups.push(group.clone());
    }
    for fact in uncovered { all_groups.push(vec![fact]); }
    all_groups
}

pub fn sort_groups(groups: Vec<Vec<String>>) -> Vec<Vec<String>> {
    let mut g = groups;
    // sort elements in each group lexicographically
    for group in &mut g { group.sort(); }
    // sort groups lexicographically by their sequence of strings (like Python repr(list))
    g.sort_by(|a, b| {
        let mut it_a = a.iter();
        let mut it_b = b.iter();
        loop {
            match (it_a.next(), it_b.next()) {
                (Some(sa), Some(sb)) => {
                    let c = sa.cmp(sb);
                    if c != std::cmp::Ordering::Equal { return c; }
                }
                (None, Some(_)) => return std::cmp::Ordering::Less,
                (Some(_), None) => return std::cmp::Ordering::Greater,
                (None, None) => return std::cmp::Ordering::Equal,
            }
        }
    });
    g
}

pub fn compute_groups(domain: &crate::translate::pddl::Domain, problem: &crate::translate::pddl::Problem, atoms: &Vec<String>, _reachable_action_params: Option<HashMap<String, Vec<Vec<String>>>>) -> (Vec<Vec<String>>, Vec<Vec<String>>, Vec<Vec<String>>) {
    // ask invariant_finder for abstract groups (with ?X placeholders)
    let groups = crate::translate::invariant_finder::get_groups(domain, problem);
    // instantiate groups using atoms set
    let reachable: HashSet<String> = atoms.iter().cloned().collect();
    let instantiated = instantiate_groups(&groups, domain, problem, &reachable);
    // Add a fallback heuristic: group grounded atoms by their first argument (object-centric)
    // but only for predicates defined in the domain (exclude numeric functions).
    let mut augmented = instantiated.clone();
    // predicate -> index map to enforce domain predicate ordering when sorting
    let mut pred_index: HashMap<String, usize> = HashMap::new();
    for (i, (pname, _)) in domain.predicates.iter().enumerate() {
        pred_index.insert(pname.clone(), i);
    }
    let func_names: HashSet<String> = domain.functions.iter().map(|(n, _)| n.clone()).collect();
    // build map first_arg -> Vec<atom> but only for atoms whose predicate is in domain.predicates
    let mut by_first: HashMap<String, Vec<String>> = HashMap::new();
    for a in atoms {
        if let Some(open) = a.find('(') {
            if let Some(close) = a.rfind(')') {
                let pred = a[..open].trim().to_string();
                if func_names.contains(&pred) { continue; }
                if !pred_index.contains_key(&pred) { continue; }
                let args = &a[open+1..close];
                let parts: Vec<&str> = args.split(',').map(|s| s.trim()).collect();
                if !parts.is_empty() {
                    let first = parts[0].to_string();
                    by_first.entry(first).or_default().push(a.clone());
                }
            }
        }
    }
    // incorporate first-arg groups if they look multi-valued and are not duplicates
    for (_first, group) in by_first.into_iter() {
        if group.len() <= 1 { continue; }
        // create deduped vector
        let mut gset: HashSet<String> = group.into_iter().collect();
        let mut gvec: Vec<String> = gset.drain().collect();
        // sort by predicate order first (domain-defined), then lexicographically
        gvec.sort_by(|a, b| {
            let a_pred = a.split('(').next().unwrap_or("").to_string();
            let b_pred = b.split('(').next().unwrap_or("").to_string();
            let ai = pred_index.get(&a_pred).cloned().unwrap_or(usize::MAX);
            let bi = pred_index.get(&b_pred).cloned().unwrap_or(usize::MAX);
            if ai != bi { return ai.cmp(&bi); }
            a.cmp(b)
        });
        // check if already present (by equality of sets)
        let mut exists = false;
        for ex in &augmented {
            let exset: HashSet<String> = ex.iter().cloned().collect();
            let gset2: HashSet<String> = gvec.iter().cloned().collect();
            if exset == gset2 { exists = true; break; }
        }
        if !exists {
            augmented.push(gvec);
        }
    }
    // now sort groups: prefer groups ordered by the first object's position in the problem
    // (so item1 groups come before item2, etc.), then fall back to lexicographic group comparison.
    let mut object_index: HashMap<String, usize> = HashMap::new();
    for (i, (name, _tp)) in problem.objects.iter().enumerate() {
        object_index.insert(name.clone(), i);
    }
    let mut groups_with_key: Vec<(Option<usize>, Vec<String>)> = Vec::new();
    for mut g in augmented.into_iter() {
        g.sort();
        // extract the first argument of the first atom if possible
        let mut key: Option<usize> = None;
        if let Some(first_atom) = g.get(0) {
            if let Some(open) = first_atom.find('(') {
                if let Some(close) = first_atom.rfind(')') {
                    let args = &first_atom[open+1..close];
                    let parts: Vec<&str> = args.split(',').map(|s| s.trim()).collect();
                    if !parts.is_empty() {
                        if let Some(&idx) = object_index.get(parts[0]) { key = Some(idx); }
                    }
                }
            }
        }
        groups_with_key.push((key, g));
    }
    groups_with_key.sort_by(|(ka, a), (kb, b)| {
        match (ka, kb) {
            (Some(ai), Some(bi)) => {
                if ai != bi { return ai.cmp(bi); }
            }
            (Some(_), None) => return std::cmp::Ordering::Less,
            (None, Some(_)) => return std::cmp::Ordering::Greater,
            (None, None) => {}
        }
        // fall back to lexicographic compare of the group contents
        a.cmp(b)
    });
    let sorted: Vec<Vec<String>> = groups_with_key.into_iter().map(|(_k,g)| g).collect();
    let mutex_groups = collect_all_mutex_groups(&sorted, &reachable);
    let chosen = choose_groups(sorted.clone(), domain, problem, &reachable, true);
    // normalize ordering inside each chosen group: lexicographic sort of atom strings
    let mut chosen_sorted = chosen.clone();
    for group in &mut chosen_sorted { group.sort(); }
    let tk = build_translation_key(&chosen_sorted);
    (chosen_sorted, mutex_groups, tk)
}
