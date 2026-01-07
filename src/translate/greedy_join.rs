//! Greedy join algorithm for splitting datalog rules.
//!
//! This module implements the greedy join algorithm from Python's greedy_join.py.
//! It takes a rule with multiple conditions and splits it into a chain of
//! binary join/project rules.

use std::collections::{HashMap, HashSet};

use crate::translate::build_model::SymAtom;

/// Tracks how many times each variable appears in a list of symbolic atoms.
#[derive(Debug)]
pub struct OccurrencesTracker {
    occurrences: HashMap<String, i32>,
}

impl OccurrencesTracker {
    /// Create a new tracker from a rule (effect + conditions)
    pub fn new(effect: &SymAtom, conditions: &[SymAtom]) -> Self {
        let mut tracker = Self {
            occurrences: HashMap::new(),
        };
        tracker.update(effect, 1);
        for cond in conditions {
            tracker.update(cond, 1);
        }
        tracker
    }
    
    /// Update occurrence counts for variables in a symbolic atom
    pub fn update(&mut self, symatom: &SymAtom, delta: i32) {
        for var in &symatom.args {
            if var.starts_with('?') {
                let count = self.occurrences.entry(var.clone()).or_insert(0);
                *count += delta;
                assert!(*count >= 0, "Negative occurrence count for {}", var);
                if *count == 0 {
                    self.occurrences.remove(var);
                }
            }
        }
    }
    
    /// Get the set of variables still in use
    pub fn variables(&self) -> HashSet<String> {
        self.occurrences.keys().cloned().collect()
    }
}

/// Cost matrix for finding optimal join pairs.
/// Cost is a tuple (unique_left, unique_right, -common) for lexicographic comparison.
#[derive(Debug)]
pub struct CostMatrix {
    joinees: Vec<SymAtom>,
    cost_matrix: Vec<Vec<(i32, i32, i32)>>,
}

impl CostMatrix {
    /// Create a new cost matrix from initial conditions
    pub fn new(conditions: &[SymAtom]) -> Self {
        let mut matrix = Self {
            joinees: Vec::new(),
            cost_matrix: Vec::new(),
        };
        for cond in conditions {
            matrix.add_entry(cond.clone());
        }
        matrix
    }
    
    /// Add a new joinee and compute costs with all existing joinees
    pub fn add_entry(&mut self, joinee: SymAtom) {
        let new_row: Vec<(i32, i32, i32)> = self.joinees
            .iter()
            .map(|other| self.compute_join_cost(&joinee, other))
            .collect();
        self.cost_matrix.push(new_row);
        self.joinees.push(joinee);
    }
    
    /// Delete an entry at the given index
    pub fn delete_entry(&mut self, index: usize) {
        // Remove column from all rows after this index
        for row in &mut self.cost_matrix[index + 1..] {
            row.remove(index);
        }
        // Remove the row
        self.cost_matrix.remove(index);
        // Remove the joinee
        self.joinees.remove(index);
    }
    
    /// Find the pair with minimum join cost
    fn find_min_pair(&self) -> (usize, usize) {
        assert!(self.joinees.len() >= 2);
        let mut min_cost = (i32::MAX, i32::MAX, i32::MAX);
        let mut left_index = 0;
        let mut right_index = 0;
        
        for (i, row) in self.cost_matrix.iter().enumerate() {
            for (j, &entry) in row.iter().enumerate() {
                if entry < min_cost {
                    min_cost = entry;
                    left_index = i;
                    right_index = j;
                }
            }
        }
        (left_index, right_index)
    }
    
    /// Remove and return the pair with minimum join cost
    pub fn remove_min_pair(&mut self) -> (SymAtom, SymAtom) {
        let (left_index, right_index) = self.find_min_pair();
        let left = self.joinees[left_index].clone();
        let right = self.joinees[right_index].clone();
        
        // Delete in order (larger index first to avoid shifting issues)
        assert!(left_index > right_index);
        self.delete_entry(left_index);
        self.delete_entry(right_index);
        
        (left, right)
    }
    
    /// Compute join cost between two symbolic atoms.
    /// Cost tuple: (unique_left, unique_right, -common) for lexicographic comparison.
    /// Lower is better: prefer joins with more shared variables and fewer unique variables.
    fn compute_join_cost(&self, left: &SymAtom, right: &SymAtom) -> (i32, i32, i32) {
        let left_vars: HashSet<_> = left.args.iter().filter(|a| a.starts_with('?')).collect();
        let right_vars: HashSet<_> = right.args.iter().filter(|a| a.starts_with('?')).collect();
        
        let (smaller, larger) = if left_vars.len() > right_vars.len() {
            (&right_vars, &left_vars)
        } else {
            (&left_vars, &right_vars)
        };
        
        let common_count = smaller.iter().filter(|v| larger.contains(*v)).count() as i32;
        
        (
            smaller.len() as i32 - common_count,
            larger.len() as i32 - common_count,
            -common_count,
        )
    }
    
    /// Check if there are at least 2 joinees
    pub fn can_join(&self) -> bool {
        self.joinees.len() >= 2
    }
}

/// Result list for building the chain of rules
pub struct ResultList {
    final_effect: SymAtom,
    result: Vec<Rule>,
    name_counter: usize,
    name_prefix: String,
}

/// A rule with type annotation
#[derive(Debug, Clone)]
pub struct Rule {
    pub rtype: String, // "project" or "join"
    pub conditions: Vec<SymAtom>,
    pub effect: SymAtom,
}

impl ResultList {
    pub fn new(effect: SymAtom, name_prefix: String) -> Self {
        Self {
            final_effect: effect,
            result: Vec::new(),
            name_counter: 0,
            name_prefix,
        }
    }
    
    /// Get the final result, updating the last rule's effect to the final effect
    pub fn get_result(mut self) -> Vec<Rule> {
        if let Some(last) = self.result.last_mut() {
            last.effect = self.final_effect;
        }
        self.result
    }
    
    /// Add a rule and return its effect as a new symbolic atom
    pub fn add_rule(&mut self, rtype: &str, conditions: Vec<SymAtom>, effect_vars: Vec<String>) -> SymAtom {
        let name = format!("{}@{}", self.name_prefix, self.name_counter);
        self.name_counter += 1;
        
        let effect = SymAtom::new(name, effect_vars);
        let rule = Rule {
            rtype: rtype.to_string(),
            conditions,
            effect: effect.clone(),
        };
        self.result.push(rule);
        effect
    }
}

/// Main greedy join algorithm.
/// 
/// Takes a rule with multiple conditions and splits it into a chain of
/// binary join/project rules using a greedy heuristic.
/// 
/// Returns a list of rules where each rule has at most 2 conditions.
pub fn greedy_join(
    effect: &SymAtom,
    conditions: &[SymAtom],
    name_prefix: &str,
) -> Vec<Rule> {
    assert!(conditions.len() >= 2, "greedy_join requires at least 2 conditions");
    
    let mut cost_matrix = CostMatrix::new(conditions);
    let mut occurrences = OccurrencesTracker::new(effect, conditions);
    let mut result = ResultList::new(effect.clone(), name_prefix.to_string());
    
    while cost_matrix.can_join() {
        // Remove pair with minimum join cost
        let (left, right) = cost_matrix.remove_min_pair();
        
        // Update occurrence counts (decrement for removed conditions)
        occurrences.update(&left, -1);
        occurrences.update(&right, -1);
        
        // Compute variables
        let left_vars: HashSet<_> = left.args.iter().filter(|a| a.starts_with('?')).cloned().collect();
        let right_vars: HashSet<_> = right.args.iter().filter(|a| a.starts_with('?')).cloned().collect();
        let common_vars: HashSet<_> = left_vars.intersection(&right_vars).cloned().collect();
        let condition_vars: HashSet<_> = left_vars.union(&right_vars).cloned().collect();
        let effect_vars: HashSet<_> = occurrences.variables()
            .intersection(&condition_vars)
            .cloned()
            .collect();
        
        // Add projection rules if needed
        let mut joinees = vec![left, right];
        for (i, joinee) in joinees.clone().iter().enumerate() {
            let joinee_vars: HashSet<_> = joinee.args.iter().filter(|a| a.starts_with('?')).cloned().collect();
            let retained_vars: HashSet<_> = joinee_vars
                .iter()
                .filter(|v| effect_vars.contains(*v) || common_vars.contains(*v))
                .cloned()
                .collect();
            
            if retained_vars != joinee_vars {
                // Need to project
                let mut sorted_vars: Vec<_> = retained_vars.into_iter().collect();
                sorted_vars.sort();
                joinees[i] = result.add_rule("project", vec![joinee.clone()], sorted_vars);
            }
        }
        
        // Create join rule
        let mut sorted_effect_vars: Vec<_> = effect_vars.into_iter().collect();
        sorted_effect_vars.sort();
        let joint_condition = result.add_rule("join", joinees, sorted_effect_vars);
        
        // Update tracking structures
        cost_matrix.add_entry(joint_condition.clone());
        occurrences.update(&joint_condition, 1);
    }
    
    result.get_result()
}

/// Get all variables from a list of symbolic atoms
pub fn get_variables(atoms: &[SymAtom]) -> HashSet<String> {
    atoms
        .iter()
        .flat_map(|a| a.args.iter())
        .filter(|a| a.starts_with('?'))
        .cloned()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_occurrences_tracker() {
        let effect = SymAtom::new("head".to_string(), vec!["?x".to_string(), "?y".to_string()]);
        let cond1 = SymAtom::new("p".to_string(), vec!["?x".to_string()]);
        let cond2 = SymAtom::new("q".to_string(), vec!["?y".to_string(), "?z".to_string()]);
        
        let tracker = OccurrencesTracker::new(&effect, &[cond1, cond2]);
        let vars = tracker.variables();
        
        assert!(vars.contains("?x"));
        assert!(vars.contains("?y"));
        assert!(vars.contains("?z"));
    }
    
    #[test]
    fn test_cost_matrix() {
        let cond1 = SymAtom::new("p".to_string(), vec!["?x".to_string()]);
        let cond2 = SymAtom::new("q".to_string(), vec!["?x".to_string(), "?y".to_string()]);
        let cond3 = SymAtom::new("r".to_string(), vec!["?z".to_string()]);
        
        let matrix = CostMatrix::new(&[cond1, cond2, cond3]);
        assert!(matrix.can_join());
        assert_eq!(matrix.joinees.len(), 3);
    }
    
    #[test]
    fn test_greedy_join_simple() {
        let effect = SymAtom::new("head".to_string(), vec!["?x".to_string(), "?y".to_string()]);
        let cond1 = SymAtom::new("p".to_string(), vec!["?x".to_string()]);
        let cond2 = SymAtom::new("q".to_string(), vec!["?x".to_string(), "?y".to_string()]);
        let cond3 = SymAtom::new("r".to_string(), vec!["?y".to_string()]);
        
        let rules = greedy_join(&effect, &[cond1, cond2, cond3], "@new-atom");
        
        // Should produce at least one join rule
        assert!(!rules.is_empty());
        
        // Last rule should have the original effect
        let last = rules.last().unwrap();
        assert_eq!(last.effect.predicate, "head");
    }
}

