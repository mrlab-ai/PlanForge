use crate::translate::build_model;
use crate::translate::split_rules::{RuleWithType, SymRule};

fn get_variables(atom: &build_model::SymAtom) -> std::collections::HashSet<String> {
    atom.args
        .iter()
        .filter(|a| a.starts_with('?'))
        .cloned()
        .collect()
}

#[derive(Clone)]
struct OccurrencesTracker {
    occurrences: std::collections::HashMap<String, i32>,
}

impl OccurrencesTracker {
    fn new(rule: &SymRule) -> Self {
        let mut tracker = Self {
            occurrences: std::collections::HashMap::new(),
        };
        tracker.update(&rule.effect, 1);
        for cond in &rule.conditions {
            tracker.update(cond, 1);
        }
        tracker
    }

    fn update(&mut self, atom: &build_model::SymAtom, delta: i32) {
        for var in atom.args.iter().filter(|a| a.starts_with('?')) {
            let entry = self.occurrences.entry(var.clone()).or_insert(0);
            *entry += delta;
            if *entry == 0 {
                self.occurrences.remove(var);
            }
        }
    }

    fn variables(&self) -> std::collections::HashSet<String> {
        self.occurrences.keys().cloned().collect()
    }
}

#[derive(Clone)]
struct CostMatrix {
    joinees: Vec<build_model::SymAtom>,
    cost_matrix: Vec<Vec<(i32, i32, i32)>>,
}

impl CostMatrix {
    fn new(joinees: Vec<build_model::SymAtom>) -> Self {
        let mut cm = Self {
            joinees: Vec::new(),
            cost_matrix: Vec::new(),
        };
        for j in joinees {
            cm.add_entry(j);
        }
        cm
    }

    fn add_entry(&mut self, joinee: build_model::SymAtom) {
        let new_row: Vec<(i32, i32, i32)> = self
            .joinees
            .iter()
            .map(|other| compute_join_cost(&joinee, other))
            .collect();
        self.cost_matrix.push(new_row);
        self.joinees.push(joinee);
    }

    fn delete_entry(&mut self, index: usize) {
        for row in self.cost_matrix.iter_mut().skip(index + 1) {
            row.remove(index);
        }
        self.cost_matrix.remove(index);
        self.joinees.remove(index);
    }

    fn find_min_pair(&self) -> (usize, usize) {
        let mut best: Option<((i32, i32, i32), usize, usize)> = None;
        for (i, row) in self.cost_matrix.iter().enumerate() {
            for (j, entry) in row.iter().enumerate() {
                if let Some((best_entry, _, _)) = &best {
                    if entry < best_entry {
                        best = Some((*entry, i, j));
                    }
                } else {
                    best = Some((*entry, i, j));
                }
            }
        }
        let (_, i, j) = best.expect("at least one pair");
        (i, j)
    }

    fn remove_min_pair(&mut self) -> (build_model::SymAtom, build_model::SymAtom) {
        let (left_index, right_index) = self.find_min_pair();
        let (li, ri) = if left_index > right_index {
            (left_index, right_index)
        } else {
            (right_index, left_index)
        };
        let left = self.joinees[li].clone();
        let right = self.joinees[ri].clone();
        self.delete_entry(li);
        self.delete_entry(ri);
        (left, right)
    }

    fn can_join(&self) -> bool {
        self.joinees.len() >= 2
    }
}

fn compute_join_cost(left: &build_model::SymAtom, right: &build_model::SymAtom) -> (i32, i32, i32) {
    let left_vars = get_variables(left);
    let right_vars = get_variables(right);
    let (small, large) = if left_vars.len() <= right_vars.len() {
        (left_vars, right_vars)
    } else {
        (right_vars, left_vars)
    };
    let common: std::collections::HashSet<String> = small.intersection(&large).cloned().collect();
    let common_len = common.len() as i32;
    (
        (small.len() as i32) - common_len,
        (large.len() as i32) - common_len,
        -common_len,
    )
}

#[derive(Clone)]
struct ResultList {
    final_effect: build_model::SymAtom,
    result: Vec<RuleWithType>,
    counter: usize,
}

impl ResultList {
    fn new(rule: &SymRule, counter: usize) -> Self {
        Self {
            final_effect: rule.effect.clone(),
            result: Vec::new(),
            counter,
        }
    }

    fn next_name(&mut self) -> String {
        let name = format!("p${}", self.counter);
        self.counter += 1;
        name
    }

    fn add_rule(
        &mut self,
        rtype: &str,
        conditions: Vec<build_model::SymAtom>,
        effect_vars: Vec<String>,
    ) -> build_model::SymAtom {
        let effect = build_model::SymAtom::new(self.next_name(), effect_vars);
        self.result.push(RuleWithType {
            rtype: rtype.to_string(),
            conditions,
            effect: effect.clone(),
        });
        effect
    }

    fn into_result(mut self) -> (Vec<RuleWithType>, usize) {
        if let Some(last) = self.result.last_mut() {
            last.effect = self.final_effect;
        }
        (self.result, self.counter)
    }
}

pub fn greedy_join(rule: &SymRule, counter: &mut usize) -> Vec<RuleWithType> {
    let mut cost_matrix = CostMatrix::new(rule.conditions.clone());
    let mut occurrences = OccurrencesTracker::new(rule);
    let mut result = ResultList::new(rule, *counter);

    while cost_matrix.can_join() {
        let (left, right) = cost_matrix.remove_min_pair();
        for joinee in [&left, &right] {
            occurrences.update(joinee, -1);
        }

        let left_vars = get_variables(&left);
        let right_vars = get_variables(&right);
        let common_vars: std::collections::HashSet<String> =
            left_vars.intersection(&right_vars).cloned().collect();
        let condition_vars: std::collections::HashSet<String> =
            left_vars.union(&right_vars).cloned().collect();
        let effect_vars_set: std::collections::HashSet<String> = occurrences
            .variables()
            .intersection(&condition_vars)
            .cloned()
            .collect();

        let mut joinees = vec![left, right];
        for joinee in joinees.iter_mut() {
            let joinee_vars = get_variables(joinee);
            let retained_vars: std::collections::HashSet<String> = joinee_vars
                .intersection(&effect_vars_set)
                .chain(joinee_vars.intersection(&common_vars))
                .cloned()
                .collect();
            if retained_vars.len() != joinee_vars.len() {
                let mut vars: Vec<String> = retained_vars.into_iter().collect();
                vars.sort();
                let proj_effect = result.add_rule("project", vec![joinee.clone()], vars);
                *joinee = proj_effect;
            }
        }

        let mut effect_vars: Vec<String> = effect_vars_set.into_iter().collect();
        effect_vars.sort();
        let joint = result.add_rule("join", joinees, effect_vars);
        cost_matrix.add_entry(joint.clone());
        occurrences.update(&joint, 1);
    }

    let (rules, new_counter) = result.into_result();
    *counter = new_counter;
    rules
}
