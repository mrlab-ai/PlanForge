use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct VariableBlock {
    pub name: String,
    pub axiom_layer: i32,
    pub range: usize,
    pub values: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct OperatorBlock {
    pub name: String,
    pub num_preconditions: usize,
    pub preconditions: Vec<String>,
    pub num_ass_effects: usize,
    pub ass_effects: Vec<String>,
    pub num_effects: usize,
    pub effects: Vec<String>,
    pub cost: f64,
}

#[derive(Debug, Clone, Default)]
pub struct SasParsedOutput {
    pub variables: HashMap<String, VariableBlock>,
    pub numeric_variables: Vec<String>,
    pub rules: Vec<Vec<String>>,
    pub comparison_axioms: Vec<String>,
    pub numeric_axioms: Vec<String>,
    pub initial_state: Vec<i32>,
    pub initial_numeric_state: Vec<f64>,
    pub operators: Vec<OperatorBlock>,
    pub goal: Vec<String>,
    pub global_constraints: Vec<String>,
    pub version: Option<i32>,
    pub metric_criterion: String,
    pub metric_index: isize,
}

#[derive(Debug, Clone)]
pub struct SimpleToRestrictedTask {
    pub parsed: SasParsedOutput,
    pub real_numeric_variables: Vec<usize>,
    pub original_real_numeric_variables: HashSet<usize>,
    pub real_var_pos: HashMap<usize, usize>,
    pub used_numeric_vars: Option<HashSet<usize>>,
    pub formulas: HashMap<usize, Vec<f64>>,
    pub mixed_real_vars_in_linear_computation: HashSet<usize>,
    pub added_constants: HashMap<String, usize>,
}

impl SimpleToRestrictedTask {
    pub fn new(parsed: SasParsedOutput) -> Self {
        let mut real_numeric_variables = Vec::new();
        for (idx, entry) in parsed.numeric_variables.iter().enumerate() {
            if entry.trim_start().starts_with('R') {
                real_numeric_variables.push(idx);
            }
        }
        let original_real_numeric_variables = real_numeric_variables.iter().copied().collect();
        let mut real_var_pos = HashMap::new();
        for (pos, var_id) in real_numeric_variables.iter().enumerate() {
            real_var_pos.insert(*var_id, pos);
        }
        Self {
            parsed,
            real_numeric_variables,
            original_real_numeric_variables,
            real_var_pos,
            used_numeric_vars: None,
            formulas: HashMap::new(),
            mixed_real_vars_in_linear_computation: HashSet::new(),
            added_constants: HashMap::new(),
        }
    }

    pub fn get_var_value(&self, var_number: usize) -> f64 {
        self.parsed.initial_numeric_state[var_number]
    }

    pub fn is_real_variable(&self, var_number: usize) -> bool {
        self.parsed.numeric_variables[var_number]
            .trim_start()
            .starts_with('R')
    }

    pub fn update_var(
        &mut self,
        var: usize,
        upd_var_number: usize,
        is_update_multiplication: bool,
    ) -> Result<(), String> {
        if self.is_real_variable(upd_var_number) {
            return Err(
                "Trying to operate on multiple real variables outside of formula".to_string(),
            );
        }
        let upd_val = self.get_var_value(upd_var_number);
        if !is_update_multiplication {
            self.parsed.initial_numeric_state[var] += upd_val;
            return Ok(());
        }

        self.parsed.initial_numeric_state[var] *= upd_val;
        let numeric_vars_snapshot = self.parsed.numeric_variables.clone();
        let initial_state_snapshot = self.parsed.initial_numeric_state.clone();

        let is_real = |idx: usize| -> bool {
            numeric_vars_snapshot
                .get(idx)
                .map(|entry| entry.trim_start().starts_with('R'))
                .unwrap_or(false)
        };

        for operator in &mut self.parsed.operators {
            for effect in &mut operator.effects {
                let parts: Vec<&str> = effect.split_whitespace().collect();
                if parts.len() < 4 {
                    continue;
                }
                let var1: usize = match parts[1].parse() {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                let var2: usize = match parts.last().unwrap().parse() {
                    Ok(v) => v,
                    Err(_) => continue,
                };

                if is_real(var1) && is_real(var2) {
                    return Err("Cannot multiply two real variables".to_string());
                }

                if var1 == var {
                    let new_index = self.parsed.numeric_variables.len();
                    self.parsed.numeric_variables.push(format!(
                        "C {} PNE derived! {} * {}",
                        new_index, var2, upd_var_number
                    ));
                    self.parsed
                        .initial_numeric_state
                        .push(initial_state_snapshot[var2] * upd_val);
                    let updated_var2 = self.parsed.initial_numeric_state.len() - 1;
                    *effect = format!("{} {} {} {}", parts[0], var1, parts[2], updated_var2);
                }
            }
        }
        Ok(())
    }

    pub fn is_var_useful(&self, var: usize) -> bool {
        if let Some(used) = &self.used_numeric_vars {
            return used.contains(&var);
        }

        let var_str = var.to_string();
        for axiom in &self.parsed.comparison_axioms {
            for token in axiom.split_whitespace().skip(1) {
                if token == var_str {
                    return true;
                }
            }
        }

        for axiom in &self.parsed.numeric_axioms {
            for token in axiom.split_whitespace().skip(1) {
                if token == var_str {
                    return true;
                }
            }
        }

        for operator in &self.parsed.operators {
            for effect in &operator.effects {
                let parts: Vec<&str> = effect.split_whitespace().collect();
                if parts.len() < 4 {
                    continue;
                }
                if let Ok(var11) = parts[1].parse::<usize>() {
                    if var11 == var {
                        return true;
                    }
                }
                if let Ok(var22) = parts.last().unwrap().parse::<usize>() {
                    if var22 == var {
                        return true;
                    }
                }
            }
        }

        false
    }

    pub fn replace_var(&mut self, var1: usize, var2: usize) {
        let var1_str = var1.to_string();
        let var2_str = var2.to_string();

        for axiom in &mut self.parsed.comparison_axioms {
            let mut tokens: Vec<String> = axiom
                .split_whitespace()
                .map(|token| token.to_string())
                .collect();
            for token in tokens.iter_mut().skip(1) {
                if *token == var1_str {
                    *token = var2_str.clone();
                }
            }
            *axiom = tokens.join(" ");
        }

        for axiom in &mut self.parsed.numeric_axioms {
            let mut tokens: Vec<String> = axiom
                .split_whitespace()
                .map(|token| token.to_string())
                .collect();
            for token in tokens.iter_mut().skip(1) {
                if *token == var1_str {
                    *token = var2_str.clone();
                }
            }
            *axiom = tokens.join(" ");
        }

        for operator in &mut self.parsed.operators {
            for effect in &mut operator.effects {
                let parts: Vec<&str> = effect.split_whitespace().collect();
                if parts.len() < 4 {
                    continue;
                }
                let mut var11 = parts[1].parse::<usize>().unwrap_or(var1);
                let mut var22 = parts.last().unwrap().parse::<usize>().unwrap_or(var1);
                if var11 == var1 {
                    var11 = var2;
                }
                if var22 == var1 {
                    var22 = var2;
                }
                *effect = format!("{} {} {} {}", parts[0], var11, parts[2], var22);
            }
        }
    }

    fn add_or_minus_formulas(&self, f1: &[f64], f2: &[f64], op: &str) -> Vec<f64> {
        let mut result = vec![0.0; f1.len()];
        if op == "+" {
            for i in 0..f1.len() {
                result[i] = f1[i] + f2[i];
            }
        } else {
            for i in 0..f1.len() {
                result[i] = f1[i] - f2[i];
            }
        }
        result
    }

    fn mult_formula(&self, f1: &[f64], f2: &[f64]) -> Vec<f64> {
        let mut result = vec![0.0; f1.len()];
        let has_real = f1.iter().skip(1).any(|val| *val != 0.0);
        if has_real {
            for i in 0..f1.len() {
                result[i] = f1[i] * f2[0];
            }
        } else {
            for i in 0..f1.len() {
                result[i] = f1[0] * f2[i];
            }
        }
        result
    }

    fn operate(&self, f1: &[f64], f2: &[f64], op: &str) -> Vec<f64> {
        if op == "*" {
            self.mult_formula(f1, f2)
        } else {
            self.add_or_minus_formulas(f1, f2, op)
        }
    }

    pub fn gen_initial_formula(&self, var: usize) -> Vec<f64> {
        let mut formula = vec![0.0; self.real_numeric_variables.len() + 1];
        if self.is_real_variable(var) {
            if let Some(pos) = self.real_var_pos.get(&var) {
                formula[pos + 1] = 1.0;
            }
        } else {
            formula[0] = self.get_var_value(var);
        }
        formula
    }

    pub fn gen_all_formulas_for_axiom(&mut self, axiom: &str) {
        let parts: Vec<&str> = axiom.split_whitespace().collect();
        if parts.len() < 4 {
            return;
        }
        let var_final: usize = match parts[0].parse() {
            Ok(v) => v,
            Err(_) => return,
        };
        let op = parts[1];
        let var1: usize = match parts[2].parse() {
            Ok(v) => v,
            Err(_) => return,
        };
        let var2: usize = match parts[3].parse() {
            Ok(v) => v,
            Err(_) => return,
        };

        let f1 = self
            .formulas
            .get(&var1)
            .cloned()
            .unwrap_or_else(|| {
                let f = self.gen_initial_formula(var1);
                f
            });
        self.formulas.entry(var1).or_insert_with(|| f1.clone());

        let f2 = self
            .formulas
            .get(&var2)
            .cloned()
            .unwrap_or_else(|| {
                let f = self.gen_initial_formula(var2);
                f
            });
        self.formulas.entry(var2).or_insert_with(|| f2.clone());

        let f3 = self.operate(&f1, &f2, op);
        self.formulas.insert(var_final, f3);
    }

    fn topological_sort_predecessors(
        &self,
        predecessors: &HashMap<usize, Vec<usize>>,
    ) -> Option<Vec<usize>> {
        let mut graph: HashMap<usize, Vec<usize>> = HashMap::new();
        let mut in_degree: HashMap<usize, usize> = HashMap::new();

        for (node, preds) in predecessors {
            in_degree.entry(*node).or_insert(0);
            for pred in preds {
                graph.entry(*pred).or_default().push(*node);
                *in_degree.entry(*node).or_insert(0) += 1;
            }
        }

        let mut queue: Vec<usize> = predecessors
            .keys()
            .filter(|node| *in_degree.get(node).unwrap_or(&0) == 0)
            .copied()
            .collect();
        let mut result = Vec::new();

        while let Some(node) = queue.pop() {
            result.push(node);
            if let Some(neigh) = graph.get(&node) {
                for next in neigh {
                    if let Some(deg) = in_degree.get_mut(next) {
                        if *deg > 0 {
                            *deg -= 1;
                            if *deg == 0 {
                                queue.push(*next);
                            }
                        }
                    }
                }
            }
        }

        if result.len() != predecessors.len() {
            None
        } else {
            Some(result)
        }
    }

    pub fn build_formulas_from_numeric_axioms(&mut self) -> Result<(), String> {
        let mut predecessors: HashMap<usize, Vec<usize>> = HashMap::new();
        let mut numeric_axiom_by_final: HashMap<usize, String> = HashMap::new();

        for axiom in &self.parsed.numeric_axioms {
            let parts: Vec<&str> = axiom.split_whitespace().collect();
            if parts.len() < 4 {
                continue;
            }
            let var_final: usize = match parts[0].parse() {
                Ok(v) => v,
                Err(_) => continue,
            };
            let var1: usize = match parts[2].parse() {
                Ok(v) => v,
                Err(_) => continue,
            };
            let var2: usize = match parts[3].parse() {
                Ok(v) => v,
                Err(_) => continue,
            };
            predecessors.insert(var_final, vec![var1, var2]);
            numeric_axiom_by_final.insert(var_final, axiom.clone());
        }

        for idx in 0..self.parsed.numeric_variables.len() {
            predecessors.entry(idx).or_insert_with(Vec::new);
        }

        let sorted_nodes = self
            .topological_sort_predecessors(&predecessors)
            .ok_or_else(|| "numeric axiom graph contains a cycle".to_string())?;

        for node in sorted_nodes {
            if let Some(axiom) = numeric_axiom_by_final.get(&node) {
                self.gen_all_formulas_for_axiom(axiom);
            }
        }

        Ok(())
    }

    pub fn mark_comparison_lhs_real(&mut self) {
        for axiom in &self.parsed.comparison_axioms {
            let parts: Vec<&str> = axiom.split_whitespace().collect();
            if parts.len() < 3 {
                continue;
            }
            let var1: usize = match parts[2].parse() {
                Ok(v) => v,
                Err(_) => continue,
            };
            if let Some(entry) = self.parsed.numeric_variables.get_mut(var1) {
                if let Some(first) = entry.chars().next() {
                    let mut new_entry = entry.clone();
                    if first != 'R' {
                        new_entry.replace_range(0..1, "R");
                        *entry = new_entry;
                    }
                }
            }
            if !self.formulas.contains_key(&var1) {
                let formula = self.gen_initial_formula(var1);
                self.formulas.insert(var1, formula);
            }
        }
    }

    pub fn update_var_with_formula(&mut self, var: usize) -> Result<(), String> {
        let formula = self
            .formulas
            .get(&var)
            .cloned()
            .unwrap_or_else(|| {
                let f = self.gen_initial_formula(var);
                f
            });
        self.formulas.entry(var).or_insert_with(|| formula.clone());

        let mut total_initial_upd_value = 0.0;
        for (i, coeff) in formula.iter().enumerate().skip(1) {
            if let Some(real_var) = self.real_numeric_variables.get(i - 1) {
                total_initial_upd_value += coeff * self.get_var_value(*real_var);
            }
        }
        self.parsed.initial_numeric_state[var] = formula[0] + total_initial_upd_value;

        let initial_state_snapshot = self.parsed.initial_numeric_state.clone();
        for operator in &mut self.parsed.operators {
            let mut total_effect_upd_value = 0.0;
            for effect in &operator.effects {
                let parts: Vec<&str> = effect.split_whitespace().collect();
                if parts.len() < 4 {
                    continue;
                }
                let var1: usize = match parts[1].parse() {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                let var2: usize = match parts[3].parse() {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                let op = parts[2];
                let pos = match self.real_var_pos.get(&var1) {
                    Some(p) => *p,
                    None => continue,
                };
                if op == "+" {
                    total_effect_upd_value += formula[pos + 1] * initial_state_snapshot[var2];
                } else if op == "-" {
                    total_effect_upd_value -= formula[pos + 1] * initial_state_snapshot[var2];
                } else {
                    let coeff = formula[pos + 1];
                    if coeff == 0.0 {
                        continue;
                    }
                    if self.mixed_real_vars_in_linear_computation.contains(&var1) {
                        return Err(format!(
                            "Unsupported assignment-like numeric effect on var {}: {}",
                            var1, effect
                        ));
                    }
                    continue;
                }
            }

            if total_effect_upd_value != 0.0 {
                operator.num_effects += 1;
                let key = format!("{:.12}", total_effect_upd_value);
                let add_idx = if let Some(idx) = self.added_constants.get(&key) {
                    *idx
                } else {
                    let new_index = self.parsed.numeric_variables.len();
                    self.parsed.numeric_variables.push(format!(
                        "C -1 !derived{}from{} : {:?}",
                        total_effect_upd_value, var, formula
                    ));
                    self.parsed.initial_numeric_state.push(total_effect_upd_value);
                    self.added_constants.insert(key, new_index);
                    new_index
                };
                operator
                    .effects
                    .push(format!("0 {} + {}", var, add_idx));
            }
        }

        Ok(())
    }

    pub fn compute_mixed_real_vars_in_linear_computation(&mut self) {
        let mut mixed = HashSet::new();

        for formula in self.formulas.values() {
            let mut nonzero_reals = Vec::new();
            for (i, coeff) in formula.iter().enumerate().skip(1) {
                if *coeff != 0.0 {
                    if let Some(real_var) = self.real_numeric_variables.get(i - 1) {
                        nonzero_reals.push(*real_var);
                    }
                }
            }
            if nonzero_reals.len() >= 2 {
                mixed.extend(nonzero_reals);
            }
        }

        for op in &self.parsed.operators {
            for eff in &op.effects {
                let parts: Vec<&str> = eff.split_whitespace().collect();
                if parts.len() < 4 {
                    continue;
                }
                let target: usize = match parts[1].parse() {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                let rhs: usize = match parts[3].parse() {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                let op = parts[2];
                if (op == "+" || op == "-")
                    && self.is_real_variable(target)
                    && self.is_real_variable(rhs)
                {
                    mixed.insert(target);
                    mixed.insert(rhs);
                }
            }
        }

        self.mixed_real_vars_in_linear_computation = mixed;
    }

    pub fn validate_assignment_like_numeric_effects(&self) -> Result<(), String> {
        for op in &self.parsed.operators {
            for eff in &op.effects {
                let parts: Vec<&str> = eff.split_whitespace().collect();
                if parts.len() < 4 {
                    continue;
                }
                let oper = parts[2];
                if oper == "+" || oper == "-" {
                    continue;
                }
                let target: usize = match parts[1].parse() {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                if !self.is_real_variable(target) {
                    continue;
                }
                if self.mixed_real_vars_in_linear_computation.contains(&target) {
                    return Err(format!(
                        "Unsupported assignment-like numeric effect: {}",
                        eff
                    ));
                }
            }
        }
        Ok(())
    }

    pub fn build_used_numeric_vars(&self) -> HashSet<usize> {
        let mut used = HashSet::new();

        for line in &self.parsed.comparison_axioms {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() == 4 {
                for pos in [2, 3] {
                    if let Ok(val) = parts[pos].parse::<usize>() {
                        used.insert(val);
                    }
                }
            }
        }

        for line in &self.parsed.numeric_axioms {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() == 4 {
                for pos in [0, 2, 3] {
                    if let Ok(val) = parts[pos].parse::<usize>() {
                        used.insert(val);
                    }
                }
            }
        }

        for op in &self.parsed.operators {
            for eff in &op.effects {
                let parts: Vec<&str> = eff.split_whitespace().collect();
                if parts.len() >= 4 {
                    if let Ok(v1) = parts[1].parse::<usize>() {
                        used.insert(v1);
                    }
                    if let Ok(v2) = parts[3].parse::<usize>() {
                        used.insert(v2);
                    }
                }
            }
        }

        used
    }

    pub fn compute_used_propositional_vars(&self) -> HashSet<usize> {
        let mut used = HashSet::new();

        for g in &self.parsed.goal {
            let parts: Vec<&str> = g.split_whitespace().collect();
            if let Some(first) = parts.first() {
                if let Ok(val) = first.parse::<usize>() {
                    used.insert(val);
                }
            }
        }

        for op in &self.parsed.operators {
            for pr in &op.preconditions {
                let parts: Vec<&str> = pr.split_whitespace().collect();
                if let Some(first) = parts.first() {
                    if let Ok(val) = first.parse::<usize>() {
                        used.insert(val);
                    }
                }
            }
        }

        for rule in &self.parsed.rules {
            if rule.is_empty() {
                continue;
            }
            let n_conds = rule[0].parse::<usize>().unwrap_or(0);
            let cond_lines = rule.iter().skip(1).take(n_conds);
            for line in cond_lines {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if let Some(first) = parts.first() {
                    if let Ok(val) = first.parse::<usize>() {
                        used.insert(val);
                    }
                }
            }
            if let Some(eff_line) = rule.get(1 + n_conds) {
                let parts: Vec<&str> = eff_line.split_whitespace().collect();
                if let Some(first) = parts.first() {
                    if let Ok(val) = first.parse::<usize>() {
                        used.insert(val);
                    }
                }
            }
        }

        used
    }

    pub fn filter_comparison_axioms_to_relevant_props(&mut self, used_prop_vars: &HashSet<usize>) {
        let mut filtered = Vec::new();
        for line in &self.parsed.comparison_axioms {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() != 4 {
                filtered.push(line.clone());
                continue;
            }
            let prop_var: usize = match parts[0].parse() {
                Ok(v) => v,
                Err(_) => {
                    filtered.push(line.clone());
                    continue;
                }
            };
            if used_prop_vars.contains(&prop_var) {
                filtered.push(line.clone());
            }
        }
        self.parsed.comparison_axioms = filtered;
    }

    pub fn prune_irrelevant_numeric_variables(&mut self) {
        let used_props = self.compute_used_propositional_vars();
        self.filter_comparison_axioms_to_relevant_props(&used_props);

        let mut required: HashSet<usize> = HashSet::new();
        if self.parsed.metric_index >= 0
            && (self.parsed.metric_index as usize) < self.parsed.numeric_variables.len()
        {
            required.insert(self.parsed.metric_index as usize);
        }

        for line in &self.parsed.comparison_axioms {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() != 4 {
                continue;
            }
            for pos in [2, 3] {
                if let Ok(val) = parts[pos].parse::<usize>() {
                    required.insert(val);
                }
            }
        }

        let mut changed = true;
        while changed {
            changed = false;
            for op in &self.parsed.operators {
                for eff in &op.effects {
                    let parts: Vec<&str> = eff.split_whitespace().collect();
                    if parts.len() < 4 {
                        continue;
                    }
                    let target: usize = match parts[1].parse() {
                        Ok(v) => v,
                        Err(_) => continue,
                    };
                    let rhs: usize = match parts[3].parse() {
                        Ok(v) => v,
                        Err(_) => continue,
                    };
                    if required.contains(&target) && !required.contains(&rhs) {
                        required.insert(rhs);
                        changed = true;
                    }
                }
            }
        }

        if required.is_empty() {
            return;
        }

        let mut kept: Vec<usize> = required
            .into_iter()
            .filter(|idx| *idx < self.parsed.numeric_variables.len())
            .collect();
        kept.sort_unstable();
        let old_to_new: HashMap<usize, usize> = kept
            .iter()
            .enumerate()
            .map(|(new, old)| (*old, new))
            .collect();

        let remap_effects = |effects: &[String], old_to_new: &HashMap<usize, usize>| {
            let mut new_effects = Vec::new();
            for line in effects {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() < 4 {
                    new_effects.push(line.clone());
                    continue;
                }
                let target: usize = match parts[1].parse() {
                    Ok(v) => v,
                    Err(_) => {
                        new_effects.push(line.clone());
                        continue;
                    }
                };
                let rhs: usize = match parts[3].parse() {
                    Ok(v) => v,
                    Err(_) => {
                        new_effects.push(line.clone());
                        continue;
                    }
                };
                if !old_to_new.contains_key(&target) {
                    continue;
                }
                if !old_to_new.contains_key(&rhs) {
                    new_effects.push(line.clone());
                    continue;
                }
                let mut remapped = parts.iter().map(|s| s.to_string()).collect::<Vec<_>>();
                remapped[1] = old_to_new[&target].to_string();
                remapped[3] = old_to_new[&rhs].to_string();
                new_effects.push(remapped.join(" "));
            }
            new_effects
        };

        let remap_comparison = |line: &str, old_to_new: &HashMap<usize, usize>| -> String {
            let mut parts: Vec<String> = line.split_whitespace().map(|s| s.to_string()).collect();
            if parts.len() != 4 {
                return line.to_string();
            }
            for pos in [2, 3] {
                if let Ok(val) = parts[pos].parse::<usize>() {
                    if let Some(new_val) = old_to_new.get(&val) {
                        parts[pos] = new_val.to_string();
                    }
                }
            }
            parts.join(" ")
        };

        let remap_numeric_axiom = |line: &str, old_to_new: &HashMap<usize, usize>| -> String {
            let mut parts: Vec<String> = line.split_whitespace().map(|s| s.to_string()).collect();
            if parts.len() != 4 {
                return line.to_string();
            }
            for pos in [0, 2, 3] {
                if let Ok(val) = parts[pos].parse::<usize>() {
                    if let Some(new_val) = old_to_new.get(&val) {
                        parts[pos] = new_val.to_string();
                    }
                }
            }
            parts.join(" ")
        };

        self.parsed.comparison_axioms = self
            .parsed
            .comparison_axioms
            .iter()
            .map(|line| remap_comparison(line, &old_to_new))
            .collect();
        self.parsed.numeric_axioms = self
            .parsed
            .numeric_axioms
            .iter()
            .map(|line| remap_numeric_axiom(line, &old_to_new))
            .collect();

        for op in &mut self.parsed.operators {
            op.effects = remap_effects(&op.effects, &old_to_new);
            op.effects.sort();
            op.effects.dedup();
            op.num_effects = op.effects.len();
            op.ass_effects.sort();
            op.ass_effects.dedup();
            op.num_ass_effects = op.ass_effects.len();
        }

        self.parsed.numeric_variables = kept
            .iter()
            .map(|idx| self.parsed.numeric_variables[*idx].clone())
            .collect();
        self.parsed.initial_numeric_state = kept
            .iter()
            .map(|idx| self.parsed.initial_numeric_state[*idx])
            .collect();
        if let Some(new_metric) = old_to_new.get(&(self.parsed.metric_index as usize)) {
            self.parsed.metric_index = *new_metric as isize;
        }
    }

    pub fn duplicate_detect_formulas(&mut self) -> HashMap<usize, usize> {
        let old_num_vars = self.parsed.numeric_variables.len();
        let mut useful: Vec<bool> = (0..old_num_vars).map(|i| self.is_var_useful(i)).collect();
        if self.parsed.metric_index >= 0
            && (self.parsed.metric_index as usize) < old_num_vars
        {
            useful[self.parsed.metric_index as usize] = true;
        }

        let mut rep_for_formula: HashMap<(String, String), usize> = HashMap::new();
        let mut old_to_rep: HashMap<usize, usize> = HashMap::new();
        let mut removed: HashSet<usize> = HashSet::new();

        for var in 0..old_num_vars {
            if var == self.parsed.metric_index as usize {
                old_to_rep.insert(var, var);
                continue;
            }
            if self.original_real_numeric_variables.contains(&var) {
                old_to_rep.insert(var, var);
                continue;
            }
            if !useful[var] {
                removed.insert(var);
                continue;
            }

            let is_new_real = self.is_real_variable(var);
            let key = if let Some(formula) = self.formulas.get(&var) {
                (
                    if is_new_real { "R" } else { "NR" }.to_string(),
                    format!("{:?}", formula),
                )
            } else {
                ("NOFORMULA".to_string(), var.to_string())
            };
            if let Some(rep) = rep_for_formula.get(&key) {
                old_to_rep.insert(var, *rep);
                removed.insert(var);
            } else {
                rep_for_formula.insert(key, var);
                old_to_rep.insert(var, var);
            }
        }

        let kept: Vec<usize> = (0..old_num_vars)
            .filter(|idx| !removed.contains(idx))
            .collect();
        let old_to_new: HashMap<usize, usize> = kept
            .iter()
            .enumerate()
            .map(|(new, old)| (*old, new))
            .collect();

        let mut index_map: HashMap<usize, usize> = HashMap::new();
        for old_idx in 0..old_num_vars {
            if let Some(rep) = old_to_rep.get(&old_idx) {
                if let Some(new_idx) = old_to_new.get(rep) {
                    index_map.insert(old_idx, *new_idx);
                }
            }
        }

        let remap_effect_line = |line: &str, index_map: &HashMap<usize, usize>| -> String {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 4 {
                let v1 = parts[1].parse::<usize>().ok();
                let v2 = parts[3].parse::<usize>().ok();
                if let (Some(v1), Some(v2)) = (v1, v2) {
                    let mut out_parts = parts.iter().map(|s| s.to_string()).collect::<Vec<_>>();
                    if let Some(new_v1) = index_map.get(&v1) {
                        out_parts[1] = new_v1.to_string();
                    }
                    if let Some(new_v2) = index_map.get(&v2) {
                        out_parts[3] = new_v2.to_string();
                    }
                    return out_parts.join(" ");
                }
            }
            line.to_string()
        };

        let remap_comparison_axiom = |line: &str, index_map: &HashMap<usize, usize>| -> String {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() != 4 {
                return line.to_string();
            }
            let v1 = parts[2].parse::<usize>().ok();
            let v2 = parts[3].parse::<usize>().ok();
            let mut out = parts.iter().map(|s| s.to_string()).collect::<Vec<_>>();
            if let Some(v1) = v1 {
                if let Some(new_v1) = index_map.get(&v1) {
                    out[2] = new_v1.to_string();
                }
            }
            if let Some(v2) = v2 {
                if let Some(new_v2) = index_map.get(&v2) {
                    out[3] = new_v2.to_string();
                }
            }
            out.join(" ")
        };

        let remap_numeric_axiom = |line: &str, index_map: &HashMap<usize, usize>| -> String {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() != 4 {
                return line.to_string();
            }
            let mut out = parts.iter().map(|s| s.to_string()).collect::<Vec<_>>();
            for pos in [0, 2, 3] {
                if let Ok(v) = parts[pos].parse::<usize>() {
                    if let Some(new_v) = index_map.get(&v) {
                        out[pos] = new_v.to_string();
                    }
                }
            }
            out.join(" ")
        };

        self.parsed.comparison_axioms = self
            .parsed
            .comparison_axioms
            .iter()
            .map(|line| remap_comparison_axiom(line, &index_map))
            .collect();
        self.parsed.numeric_axioms = self
            .parsed
            .numeric_axioms
            .iter()
            .map(|line| remap_numeric_axiom(line, &index_map))
            .collect();

        for op in &mut self.parsed.operators {
            op.effects = op
                .effects
                .iter()
                .map(|e| remap_effect_line(e, &index_map))
                .collect();
        }

        self.parsed.numeric_variables = kept
            .iter()
            .map(|idx| self.parsed.numeric_variables[*idx].clone())
            .collect();
        self.parsed.initial_numeric_state = kept
            .iter()
            .map(|idx| self.parsed.initial_numeric_state[*idx])
            .collect();

        index_map
    }

    pub fn run_transforms(&mut self) -> Result<(), String> {
        self.build_formulas_from_numeric_axioms()?;
        self.mark_comparison_lhs_real();
        self.compute_mixed_real_vars_in_linear_computation();
        self.validate_assignment_like_numeric_effects()?;

        for idx in 0..self.parsed.numeric_variables.len() {
            if self.is_real_variable(idx) && !self.real_numeric_variables.contains(&idx) {
                self.update_var_with_formula(idx)?;
            }
        }

        self.parsed.numeric_axioms.clear();
        self.used_numeric_vars = Some(self.build_used_numeric_vars());
        let index_map = self.duplicate_detect_formulas();
        if let Some(new_metric) = index_map.get(&(self.parsed.metric_index as usize)) {
            self.parsed.metric_index = *new_metric as isize;
        }

        for op in &mut self.parsed.operators {
            op.effects.sort();
            op.effects.dedup();
            op.num_effects = op.effects.len();
            op.ass_effects.sort();
            op.ass_effects.dedup();
            op.num_ass_effects = op.ass_effects.len();
        }

        self.prune_irrelevant_numeric_variables();
        Ok(())
    }
}

pub fn parse_file(path: &Path) -> Result<SasParsedOutput, String> {
    let contents = fs::read_to_string(path)
        .map_err(|err| format!("failed to read SAS file {}: {}", path.display(), err))?;
    parse_sas_output(&contents)
}

pub fn parse_sas_output(sas_output: &str) -> Result<SasParsedOutput, String> {
    let mut parsed = SasParsedOutput::default();
    let mut lines = sas_output.lines().peekable();
    parsed.metric_criterion = "0".to_string();
    parsed.metric_index = 0;

    while let Some(line) = lines.next() {
        match line.trim() {
            "begin_variable" => {
                let name = lines
                    .next()
                    .ok_or_else(|| "missing variable name".to_string())?
                    .trim()
                    .to_string();
                let axiom_layer: i32 = lines
                    .next()
                    .ok_or_else(|| "missing axiom layer".to_string())?
                    .trim()
                    .parse()
                    .map_err(|err| format!("invalid axiom layer: {}", err))?;
                let range: usize = lines
                    .next()
                    .ok_or_else(|| "missing variable range".to_string())?
                    .trim()
                    .parse()
                    .map_err(|err| format!("invalid variable range: {}", err))?;
                let mut values = Vec::new();
                for _ in 0..range {
                    let value = lines
                        .next()
                        .ok_or_else(|| "missing variable value".to_string())?
                        .trim()
                        .to_string();
                    values.push(value);
                }
                let end = lines
                    .next()
                    .ok_or_else(|| "missing end_variable".to_string())?;
                if end.trim() != "end_variable" {
                    return Err("expected end_variable".to_string());
                }
                parsed.variables.insert(
                    name.clone(),
                    VariableBlock {
                        name,
                        axiom_layer,
                        range,
                        values,
                    },
                );
            }
            "begin_numeric_variables" => {
                while let Some(next_line) = lines.peek() {
                    if next_line.trim() == "end_numeric_variables" {
                        lines.next();
                        break;
                    }
                    parsed.numeric_variables.push(lines.next().unwrap().trim().to_string());
                }
            }
            "begin_rule" => {
                let mut rule_lines = Vec::new();
                while let Some(next_line) = lines.next() {
                    if next_line.trim() == "end_rule" {
                        break;
                    }
                    rule_lines.push(next_line.trim().to_string());
                }
                parsed.rules.push(rule_lines);
            }
            "begin_comparison_axioms" => {
                while let Some(next_line) = lines.peek() {
                    if next_line.trim() == "end_comparison_axioms" {
                        lines.next();
                        break;
                    }
                    parsed.comparison_axioms.push(lines.next().unwrap().trim().to_string());
                }
            }
            "begin_numeric_axioms" => {
                while let Some(next_line) = lines.peek() {
                    if next_line.trim() == "end_numeric_axioms" {
                        lines.next();
                        break;
                    }
                    parsed.numeric_axioms.push(lines.next().unwrap().trim().to_string());
                }
            }
            "begin_state" => {
                while let Some(next_line) = lines.peek() {
                    if next_line.trim() == "end_state" {
                        lines.next();
                        break;
                    }
                    let value: i32 = lines
                        .next()
                        .unwrap()
                        .trim()
                        .parse()
                        .map_err(|err| format!("invalid state value: {}", err))?;
                    parsed.initial_state.push(value);
                }
            }
            "begin_numeric_state" => {
                while let Some(next_line) = lines.peek() {
                    if next_line.trim() == "end_numeric_state" {
                        lines.next();
                        break;
                    }
                    let value: f64 = lines
                        .next()
                        .unwrap()
                        .trim()
                        .parse()
                        .map_err(|err| format!("invalid numeric state value: {}", err))?;
                    parsed.initial_numeric_state.push(value);
                }
            }
            "begin_operator" => {
                let name = lines
                    .next()
                    .ok_or_else(|| "missing operator name".to_string())?
                    .trim()
                    .to_string();
                let num_preconditions: usize = lines
                    .next()
                    .ok_or_else(|| "missing operator precondition count".to_string())?
                    .trim()
                    .parse()
                    .map_err(|err| format!("invalid precondition count: {}", err))?;
                let mut preconditions = Vec::new();
                for _ in 0..num_preconditions {
                    preconditions.push(
                        lines
                            .next()
                            .ok_or_else(|| "missing precondition".to_string())?
                            .trim()
                            .to_string(),
                    );
                }
                let num_ass_effects: usize = lines
                    .next()
                    .ok_or_else(|| "missing assignment effect count".to_string())?
                    .trim()
                    .parse()
                    .map_err(|err| format!("invalid assignment effect count: {}", err))?;
                let mut ass_effects = Vec::new();
                for _ in 0..num_ass_effects {
                    ass_effects.push(
                        lines
                            .next()
                            .ok_or_else(|| "missing assignment effect".to_string())?
                            .trim()
                            .to_string(),
                    );
                }
                let num_effects: usize = lines
                    .next()
                    .ok_or_else(|| "missing effect count".to_string())?
                    .trim()
                    .parse()
                    .map_err(|err| format!("invalid effect count: {}", err))?;
                let mut effects = Vec::new();
                for _ in 0..num_effects {
                    effects.push(
                        lines
                            .next()
                            .ok_or_else(|| "missing effect".to_string())?
                            .trim()
                            .to_string(),
                    );
                }
                let cost_line = lines
                    .next()
                    .ok_or_else(|| "missing operator cost".to_string())?;
                let cost: f64 = cost_line
                    .trim()
                    .parse()
                    .map_err(|err| format!("invalid operator cost: {}", err))?;
                let end = lines
                    .next()
                    .ok_or_else(|| "missing end_operator".to_string())?;
                if end.trim() != "end_operator" {
                    return Err("expected end_operator".to_string());
                }

                parsed.operators.push(OperatorBlock {
                    name,
                    num_preconditions,
                    preconditions,
                    num_ass_effects,
                    ass_effects,
                    num_effects,
                    effects,
                    cost,
                });
            }
            "begin_goal" => {
                while let Some(next_line) = lines.peek() {
                    if next_line.trim() == "end_goal" {
                        lines.next();
                        break;
                    }
                    parsed.goal.push(lines.next().unwrap().trim().to_string());
                }
            }
            "begin_global_constraint" => {
                while let Some(next_line) = lines.peek() {
                    if next_line.trim() == "end_global_constraint" {
                        lines.next();
                        break;
                    }
                    parsed
                        .global_constraints
                        .push(lines.next().unwrap().trim().to_string());
                }
            }
            "begin_version" => {
                if let Some(version_line) = lines.next() {
                    let version = version_line.trim().parse::<i32>().ok();
                    parsed.version = version;
                }
                let _ = lines.next();
            }
            "begin_metric" => {
                let mut metric_lines = Vec::new();
                while let Some(next_line) = lines.peek() {
                    if next_line.trim() == "end_metric" {
                        lines.next();
                        break;
                    }
                    metric_lines.push(lines.next().unwrap().trim().to_string());
                }
                let metric = metric_lines.join(" ");
                let mut parts = metric.split_whitespace();
                if let Some(crit) = parts.next() {
                    parsed.metric_criterion = crit.to_string();
                }
                if let Some(idx) = parts.next() {
                    parsed.metric_index = idx.parse::<isize>().unwrap_or(0);
                }
            }
            _ => {}
        }
    }

    Ok(parsed)
}
