use super::pddl_to_prolog::{get_variables, Rule, RuleType};
/// Port of greedy_join.py
/// Greedy algorithm for splitting rules into binary joins.
use std::collections::{HashMap, HashSet};

/// Python: class OccurrencesTracker(object)
struct OccurrencesTracker {
    occurrences: HashMap<String, usize>,
}

impl OccurrencesTracker {
    fn new(rule: &Rule) -> Self {
        let mut occurrences = HashMap::new();
        for arg in &rule.effect[1..] {
            if arg.starts_with('?') {
                *occurrences.entry(arg.clone()).or_insert(0) += 1;
            }
        }
        for cond in &rule.conditions {
            for arg in &cond[1..] {
                if arg.starts_with('?') {
                    *occurrences.entry(arg.clone()).or_insert(0) += 1;
                }
            }
        }
        OccurrencesTracker { occurrences }
    }

    fn update(&mut self, atom: &[String], delta: i32) {
        for arg in &atom[1..] {
            if arg.starts_with('?') {
                let entry = self.occurrences.entry(arg.clone()).or_insert(0);
                *entry = (*entry as i32 + delta) as usize;
                if *entry == 0 {
                    self.occurrences.remove(arg);
                }
            }
        }
    }

    fn variables(&self) -> HashSet<String> {
        self.occurrences.keys().cloned().collect()
    }
}

/// Python: class CostMatrix(object)
struct CostMatrix {
    joinees: Vec<Vec<String>>,
    cost_matrix: Vec<Vec<(usize, usize, i32)>>,
}

impl CostMatrix {
    fn new(joinees: Vec<Vec<String>>) -> Self {
        let mut cm = CostMatrix {
            joinees: vec![],
            cost_matrix: vec![],
        };
        for joinee in joinees {
            cm.add_entry(joinee);
        }
        cm
    }

    fn add_entry(&mut self, joinee: Vec<String>) {
        let new_row: Vec<(usize, usize, i32)> = self
            .joinees
            .iter()
            .map(|other| Self::compute_join_cost(&joinee, other))
            .collect();
        self.cost_matrix.push(new_row);
        self.joinees.push(joinee);
    }

    fn delete_entry(&mut self, index: usize) {
        for row in &mut self.cost_matrix[(index + 1)..] {
            row.remove(index);
        }
        self.cost_matrix.remove(index);
        self.joinees.remove(index);
    }

    fn find_min_pair(&self) -> (usize, usize) {
        assert!(self.joinees.len() >= 2);
        let mut min_cost = (usize::MAX, usize::MAX, 0i32);
        let mut left_index = 0;
        let mut right_index = 0;
        for (i, row) in self.cost_matrix.iter().enumerate() {
            for (j, entry) in row.iter().enumerate() {
                if *entry < min_cost {
                    min_cost = *entry;
                    left_index = i;
                    right_index = j;
                }
            }
        }
        (left_index, right_index)
    }

    fn remove_min_pair(&mut self) -> (Vec<String>, Vec<String>) {
        let (left_index, right_index) = self.find_min_pair();
        let left = self.joinees[left_index].clone();
        let right = self.joinees[right_index].clone();
        assert!(left_index > right_index);
        self.delete_entry(left_index);
        self.delete_entry(right_index);
        (left, right)
    }

    fn compute_join_cost(left: &[String], right: &[String]) -> (usize, usize, i32) {
        let left_vars = get_variables(&[left.to_vec()]);
        let right_vars = get_variables(&[right.to_vec()]);
        let (left_vars, right_vars) = if left_vars.len() > right_vars.len() {
            (right_vars, left_vars)
        } else {
            (left_vars, right_vars)
        };
        let common = left_vars.intersection(&right_vars).count();
        (
            left_vars.len() - common,
            right_vars.len() - common,
            -(common as i32),
        )
    }

    fn can_join(&self) -> bool {
        self.joinees.len() >= 2
    }
}

/// Python: class ResultList(object)
struct ResultList {
    final_effect: Vec<String>,
    result: Vec<Rule>,
    counter: usize,
}

impl ResultList {
    fn new(rule: &Rule, counter: usize) -> Self {
        ResultList {
            final_effect: rule.effect.clone(),
            result: vec![],
            counter,
        }
    }

    fn get_result(mut self) -> (Vec<Rule>, usize) {
        if let Some(last) = self.result.last_mut() {
            last.effect = self.final_effect;
        }
        (self.result, self.counter)
    }

    fn add_rule(
        &mut self,
        rule_type: RuleType,
        conditions: Vec<Vec<String>>,
        effect_vars: Vec<String>,
    ) -> Vec<String> {
        let pred = format!("p${}", self.counter);
        self.counter += 1;
        let mut effect = vec![pred];
        effect.extend(effect_vars);
        let rule = Rule::new_typed(conditions, effect.clone(), rule_type);
        self.result.push(rule);
        effect
    }
}

/// Python: def greedy_join(rule, name_generator)
pub fn greedy_join(rule: &Rule, counter: &mut usize) -> Vec<Rule> {
    assert!(rule.conditions.len() >= 2);

    let mut cost_matrix = CostMatrix::new(rule.conditions.clone());
    let mut occurrences = OccurrencesTracker::new(rule);
    let mut result_list = ResultList::new(rule, *counter);

    while cost_matrix.can_join() {
        let (left, right) = cost_matrix.remove_min_pair();
        occurrences.update(&left, -1);
        occurrences.update(&right, -1);

        let left_vars = get_variables(&[left.clone()]);
        let right_vars = get_variables(&[right.clone()]);
        let common_vars: HashSet<String> = left_vars.intersection(&right_vars).cloned().collect();
        let condition_vars: HashSet<String> = left_vars.union(&right_vars).cloned().collect();
        let effect_vars: HashSet<String> = occurrences
            .variables()
            .intersection(&condition_vars)
            .cloned()
            .collect();

        let mut joinees = vec![left, right];
        for joinee in joinees.iter_mut() {
            let joinee_vars = get_variables(&[joinee.clone()]);
            let retained_vars: HashSet<String> = joinee_vars
                .intersection(&effect_vars.union(&common_vars).cloned().collect())
                .cloned()
                .collect();
            if retained_vars != joinee_vars {
                let mut sorted_retained: Vec<String> = retained_vars.into_iter().collect();
                sorted_retained.sort();
                *joinee =
                    result_list.add_rule(RuleType::Project, vec![joinee.clone()], sorted_retained);
            }
        }

        let mut sorted_effect_vars: Vec<String> = effect_vars.into_iter().collect();
        sorted_effect_vars.sort();
        let joint_condition = result_list.add_rule(RuleType::Join, joinees, sorted_effect_vars);
        cost_matrix.add_entry(joint_condition.clone());
        occurrences.update(&joint_condition, 1);
    }

    let (result, new_counter) = result_list.get_result();
    *counter = new_counter;
    result
}
