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
