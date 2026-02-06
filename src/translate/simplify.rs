use std::collections::{HashMap, HashSet};

use crate::translate::sas::{
    CanonicalAssignEffect, CanonicalAssignRhs, CanonicalEffect, CanonicalOperator,
    CanonicalVariable, CompareAxiom, SASAxiom, SASOperator, SASTask, Variable,
};

#[derive(Debug)]
pub enum SimplifyError {
    Impossible,
    TriviallySolvable,
}

#[derive(Debug, Clone)]
struct DomainTransitionGraph {
    init: usize,
    size: usize,
    arcs: HashMap<usize, HashSet<usize>>,
}

impl DomainTransitionGraph {
    fn new(init: usize, size: usize) -> Self {
        Self {
            init,
            size,
            arcs: HashMap::new(),
        }
    }

    fn add_arc(&mut self, u: usize, v: usize) {
        self.arcs.entry(u).or_default().insert(v);
    }

    fn reachable(&self) -> HashSet<usize> {
        let mut queue = vec![self.init];
        let mut reachable: HashSet<usize> = queue.iter().copied().collect();
        while let Some(node) = queue.pop() {
            if let Some(neigh) = self.arcs.get(&node) {
                for &n in neigh {
                    if reachable.insert(n) {
                        queue.push(n);
                    }
                }
            }
        }
        reachable
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NewValue {
    Value(usize),
    AlwaysFalse,
    AlwaysTrue,
}

#[derive(Debug)]
struct VarValueRenaming {
    new_var_nos: Vec<Option<usize>>,
    new_values: Vec<Vec<NewValue>>,
    new_sizes: Vec<usize>,
    new_var_count: usize,
    num_removed_values: usize,
}

impl VarValueRenaming {
    fn new() -> Self {
        Self {
            new_var_nos: Vec::new(),
            new_values: Vec::new(),
            new_sizes: Vec::new(),
            new_var_count: 0,
            num_removed_values: 0,
        }
    }

    fn register_variable(&mut self, old_domain_size: usize, init_value: usize, new_domain: HashSet<usize>) {
        if new_domain.len() == 1 {
            let mut new_values_for_var = vec![NewValue::AlwaysFalse; old_domain_size];
            new_values_for_var[init_value] = NewValue::AlwaysTrue;
            self.new_var_nos.push(None);
            self.new_values.push(new_values_for_var);
            self.num_removed_values += old_domain_size;
        } else {
            let mut new_value_counter = 0usize;
            let mut new_values_for_var = Vec::with_capacity(old_domain_size);
            for value in 0..old_domain_size {
                if new_domain.contains(&value) {
                    new_values_for_var.push(NewValue::Value(new_value_counter));
                    new_value_counter += 1;
                } else {
                    self.num_removed_values += 1;
                    new_values_for_var.push(NewValue::AlwaysFalse);
                }
            }
            let new_size = new_value_counter;
            self.new_var_nos.push(Some(self.new_var_count));
            self.new_values.push(new_values_for_var);
            self.new_sizes.push(new_size);
            self.new_var_count += 1;
        }
    }

    fn translate_pair(&self, fact_pair: (usize, usize)) -> (Option<usize>, NewValue) {
        let (var_no, value) = fact_pair;
        let new_var_no = self.new_var_nos[var_no];
        let new_value = self.new_values[var_no][value];
        (new_var_no, new_value)
    }

    fn convert_pairs(&self, pairs: &mut Vec<(usize, usize)>) -> Result<(), SimplifyError> {
        let mut new_pairs = Vec::new();
        for pair in pairs.iter().copied() {
            let (new_var_no, new_value) = self.translate_pair(pair);
            match new_value {
                NewValue::AlwaysFalse => return Err(SimplifyError::Impossible),
                NewValue::AlwaysTrue => {}
                NewValue::Value(v) => {
                    let new_var_no = new_var_no.expect("missing new var for kept value");
                    new_pairs.push((new_var_no, v));
                }
            }
        }
        *pairs = new_pairs;
        Ok(())
    }

    fn apply_to_task(&self, task: &mut SASTask) -> Result<(), SimplifyError> {
        self.apply_to_variables(task);
        self.apply_to_translation_key(task);
        self.apply_to_mutexes(&mut task.mutex_groups);
        self.apply_to_init(task)?;
        self.apply_to_goals(&mut task.goal)?;
        task.global_constraint = self.translate_global_constraint(task.global_constraint)?;
        self.apply_to_operators(&mut task.operators)?;
        self.apply_to_axioms(&mut task.axioms, &mut task.comparison_axioms)?;
        self.apply_to_ranges_and_layers(task);
        rebuild_canonical(task);
        Ok(())
    }

    fn apply_to_variables(&self, task: &mut SASTask) {
        let mut new_vars: Vec<Variable> = Vec::with_capacity(self.new_var_count);
        for (old_no, var) in task.variables.iter().enumerate() {
            if let Some(new_no) = self.new_var_nos[old_no] {
                if new_vars.len() <= new_no {
                    new_vars.resize_with(new_no + 1, || Variable { value_names: vec![] });
                }
                new_vars[new_no] = Variable {
                    value_names: var.value_names.clone(),
                };
            }
        }
        for (old_no, values) in task.variables.iter().enumerate() {
            if let Some(new_no) = self.new_var_nos[old_no] {
                let mut new_value_names = vec![String::new(); self.new_sizes[new_no]];
                for (old_value, name) in values.value_names.iter().enumerate() {
                    let (_nv, mapped) = self.translate_pair((old_no, old_value));
                    if let NewValue::Value(v) = mapped {
                        new_value_names[v] = name.clone();
                    }
                }
                new_vars[new_no].value_names = new_value_names;
            }
        }
        task.variables = new_vars;
    }

    fn apply_to_translation_key(&self, task: &mut SASTask) {
        let mut new_key: Vec<Vec<String>> = vec![Vec::new(); self.new_var_count];
        for (old_no, values) in task.translation_key.iter().enumerate() {
            if let Some(new_no) = self.new_var_nos[old_no] {
                let mut new_values = vec![String::new(); self.new_sizes[new_no]];
                for (old_value, name) in values.iter().enumerate() {
                    let (_nv, mapped) = self.translate_pair((old_no, old_value));
                    if let NewValue::Value(v) = mapped {
                        new_values[v] = name.clone();
                    }
                }
                new_key[new_no] = new_values;
            }
        }
        task.translation_key = new_key;
    }

    fn apply_to_ranges_and_layers(&self, task: &mut SASTask) {
        task.ranges = self.new_sizes.clone();
        let mut new_layers = vec![0; self.new_var_count];
        for (old_no, new_no) in self.new_var_nos.iter().enumerate() {
            if let Some(nn) = new_no {
                if let Some(layer) = task.axiom_layers.get(old_no) {
                    new_layers[*nn] = *layer;
                }
            }
        }
        task.axiom_layers = new_layers;
    }

    fn apply_to_mutexes(&self, mutexes: &mut Vec<Vec<(usize, usize)>>) {
        let mut new_mutexes = Vec::new();
        for group in mutexes.iter() {
            let mut new_group = Vec::new();
            for &(var, val) in group {
                let (new_var, new_val) = self.translate_pair((var, val));
                if let (Some(nv), NewValue::Value(v)) = (new_var, new_val) {
                    new_group.push((nv, v));
                }
            }
            if new_group.len() >= 2 {
                new_mutexes.push(new_group);
            }
        }
        *mutexes = new_mutexes;
    }

    fn apply_to_init(&self, task: &mut SASTask) -> Result<(), SimplifyError> {
        let mut init_pairs: Vec<(usize, usize)> = task
            .init
            .iter()
            .enumerate()
            .map(|(v, val)| (v, *val as usize))
            .collect();
        self.convert_pairs(&mut init_pairs)?;
        let mut new_values = vec![0; self.new_var_count];
        for (new_var, new_val) in init_pairs {
            new_values[new_var] = new_val as i32;
        }
        task.init = new_values;
        Ok(())
    }

    fn apply_to_goals(&self, goals: &mut Vec<(usize, usize)>) -> Result<(), SimplifyError> {
        self.convert_pairs(goals)?;
        if goals.is_empty() {
            return Err(SimplifyError::TriviallySolvable);
        }
        Ok(())
    }

    fn translate_global_constraint(
        &self,
        constraint: Option<(usize, usize)>,
    ) -> Result<Option<(usize, usize)>, SimplifyError> {
        match constraint {
            None => Ok(None),
            Some(pair) => {
                let (new_var, new_val) = self.translate_pair(pair);
                match new_val {
                    NewValue::AlwaysFalse => Err(SimplifyError::Impossible),
                    NewValue::AlwaysTrue => Ok(None),
                    NewValue::Value(v) => {
                        let nv = new_var.expect("missing new var for global constraint");
                        Ok(Some((nv, v)))
                    }
                }
            }
        }
    }

    fn apply_to_operators(&self, operators: &mut Vec<SASOperator>) -> Result<(), SimplifyError> {
        let mut new_ops = Vec::new();
        for op in operators.iter() {
            if let Some(new_op) = self.translate_operator(op)? {
                new_ops.push(new_op);
            }
        }
        *operators = new_ops;
        Ok(())
    }

    fn apply_to_axioms(
        &self,
        axioms: &mut Vec<SASAxiom>,
        comp_axioms: &mut Vec<CompareAxiom>,
    ) -> Result<(), SimplifyError> {
        let mut new_axioms = Vec::new();
        for ax in axioms.iter() {
            match self.translate_axiom(ax)? {
                Some(ax) => new_axioms.push(ax),
                None => {}
            }
        }
        *axioms = new_axioms;
        for cax in comp_axioms.iter_mut() {
            if let Some(new_var) = self.new_var_nos[cax.effect_var] {
                cax.effect_var = new_var;
            }
        }
        Ok(())
    }

    fn translate_axiom(&self, axiom: &SASAxiom) -> Result<Option<SASAxiom>, SimplifyError> {
        let mut condition = axiom.condition.clone();
        self.convert_pairs(&mut condition)?;
        let (new_var, new_val) = self.translate_pair(axiom.effect);
        match new_val {
            NewValue::AlwaysFalse => Err(SimplifyError::Impossible),
            NewValue::AlwaysTrue => Ok(None),
            NewValue::Value(v) => {
                let nv = new_var.expect("missing new var for axiom");
                Ok(Some(SASAxiom {
                    condition,
                    effect: (nv, v),
                }))
            }
        }
    }

    fn translate_operator(&self, op: &SASOperator) -> Result<Option<SASOperator>, SimplifyError> {
        let mut applicability_conditions = get_applicability_conditions(op);
        if self.convert_pairs(&mut applicability_conditions).is_err() {
            return Ok(None);
        }
        let mut conditions_dict: HashMap<usize, usize> = HashMap::new();
        for (v, val) in applicability_conditions.iter().copied() {
            conditions_dict.insert(v, val);
        }
        let mut new_prevail_vars: HashSet<usize> = conditions_dict.keys().copied().collect();
        let mut new_pre_post = Vec::new();
        for entry in op.effects.iter() {
            if let Some(new_entry) = self.translate_pre_post(entry, &conditions_dict)? {
                new_prevail_vars.remove(&new_entry.0);
                new_pre_post.push(new_entry);
            }
        }
        let mut new_assign_effects = Vec::new();
        for entry in op.numeric_effects.iter() {
            if let Some(new_entry) = self.translate_assign_effect(entry)? {
                new_assign_effects.push(new_entry);
            }
        }
        if new_pre_post.is_empty() && new_assign_effects.is_empty() {
            return Ok(None);
        }
        let mut new_prevail: Vec<(usize, usize)> = conditions_dict
            .into_iter()
            .filter(|(v, _)| new_prevail_vars.contains(v))
            .collect();
        new_prevail.sort();
        Ok(Some(SASOperator {
            name: op.name.clone(),
            prevails: new_prevail,
            effects: new_pre_post,
            numeric_effects: new_assign_effects,
            cost: op.cost,
        }))
    }

    fn translate_pre_post(
        &self,
        pre_post: &(usize, usize, usize, Vec<(usize, usize)>),
        conditions_dict: &HashMap<usize, usize>,
    ) -> Result<Option<(usize, usize, usize, Vec<(usize, usize)>)>, SimplifyError> {
        let (var_no, pre, post, cond) = pre_post;
        let (new_var_no, new_post) = self.translate_pair((*var_no, *post));
        let new_post = match new_post {
            NewValue::AlwaysTrue => return Ok(None),
            NewValue::AlwaysFalse => return Ok(None),
            NewValue::Value(v) => v,
        };
        let new_pre = match self.translate_pair((*var_no, *pre)).1 {
            NewValue::AlwaysFalse => return Ok(None),
            NewValue::AlwaysTrue => return Ok(None),
            NewValue::Value(v) => v,
        };
        if new_pre == new_post {
            return Ok(None);
        }
        let mut new_cond = cond.clone();
        if self.convert_pairs(&mut new_cond).is_err() {
            return Ok(None);
        }
        for (cond_var, cond_value) in new_cond.iter().copied() {
            if let Some(existing) = conditions_dict.get(&cond_var) {
                if *existing != cond_value {
                    return Ok(None);
                }
            }
        }
        let new_var_no = new_var_no.expect("missing new var for pre_post");
        Ok(Some((new_var_no, new_pre, new_post, new_cond)))
    }

    fn translate_assign_effect(
        &self,
        assign_effect: &(usize, String, usize, Vec<(usize, usize)>),
    ) -> Result<Option<(usize, String, usize, Vec<(usize, usize)>)>, SimplifyError> {
        let (nvar, op, rhs, cond) = assign_effect;
        let mut new_cond = cond.clone();
        if self.convert_pairs(&mut new_cond).is_err() {
            return Ok(None);
        }
        Ok(Some((*nvar, op.clone(), *rhs, new_cond)))
    }
}

fn build_dtgs(task: &SASTask) -> Vec<DomainTransitionGraph> {
    let init_vals = &task.init;
    let sizes = &task.ranges;
    let mut dtgs = Vec::new();
    for (init, size) in init_vals.iter().zip(sizes.iter()) {
        if *size > 0 {
            dtgs.push(DomainTransitionGraph::new(*init as usize, *size));
        }
    }

    let add_arc = |dtgs: &mut [DomainTransitionGraph], var_no: usize, pre_spec: Option<usize>, post: usize| {
        let pre_values: Vec<usize> = match pre_spec {
            None => (0..sizes[var_no]).filter(|v| *v != post).collect(),
            Some(p) => vec![p],
        };
        for pre in pre_values {
            if let Some(dtg) = dtgs.get_mut(var_no) {
                dtg.add_arc(pre, post);
            }
        }
    };

    for op in &task.operators {
        let conditions = get_applicability_conditions(op);
        for (var_no, _pre, post, cond) in &op.effects {
            let effective_pre = get_effective_pre(*var_no, &conditions, cond);
            if let Some(pre) = effective_pre {
                add_arc(&mut dtgs, *var_no, pre, *post);
            }
        }
    }
    for axiom in &task.axioms {
        let (var_no, val) = axiom.effect;
        add_arc(&mut dtgs, var_no, None, val);
    }
    for cax in &task.comparison_axioms {
        add_arc(&mut dtgs, cax.effect_var, None, 0);
        add_arc(&mut dtgs, cax.effect_var, None, 1);
    }

    dtgs
}

fn get_effective_pre(
    var_no: usize,
    conditions: &[(usize, usize)],
    effect_conditions: &[(usize, usize)],
) -> Option<Option<usize>> {
    let mut result: Option<usize> = conditions
        .iter()
        .find(|(v, _)| *v == var_no)
        .map(|(_, val)| *val);
    for (cond_var_no, cond_val) in effect_conditions {
        if *cond_var_no == var_no {
            if result.is_none() {
                result = Some(*cond_val);
            } else if result != Some(*cond_val) {
                return None;
            }
        }
    }
    Some(result)
}

fn get_applicability_conditions(op: &SASOperator) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    out.extend(op.prevails.iter().copied());
    for (var, pre, _post, _cond) in &op.effects {
        out.push((*var, *pre));
    }
    out
}

fn build_renaming(dtgs: &[DomainTransitionGraph]) -> VarValueRenaming {
    let mut renaming = VarValueRenaming::new();
    for dtg in dtgs {
        renaming.register_variable(dtg.size, dtg.init, dtg.reachable());
    }
    renaming
}

fn rebuild_canonical(task: &mut SASTask) {
    let axiom_var_set: HashSet<usize> = task.axioms.iter().map(|a| a.effect.0).collect();
    let comp_var_set: HashSet<usize> = task
        .comparison_axioms
        .iter()
        .map(|c| c.effect_var)
        .collect();
    task.canonical_variables = task
        .variables
        .iter()
        .enumerate()
        .map(|(idx, v)| CanonicalVariable {
            name: format!("var{}", idx),
            axiom_layer: if axiom_var_set.contains(&idx) {
                30
            } else if comp_var_set.contains(&idx) {
                29
            } else {
                -1
            },
            values: v.value_names.clone(),
        })
        .collect();
    task.canonical_operators = task
        .operators
        .iter()
        .map(|op| {
            let pre_post = op
                .effects
                .iter()
                .map(|(var, pre, post, cond)| CanonicalEffect {
                    var: *var,
                    pre: Some(*pre),
                    post: *post,
                    condition: cond.clone(),
                })
                .collect();
            let assign_effects = op
                .numeric_effects
                .iter()
                .map(|(target, assign_op, rhs_var, cond)| CanonicalAssignEffect {
                    target: *target,
                    op: assign_op.clone(),
                    rhs: CanonicalAssignRhs::Variable(*rhs_var),
                    condition: cond.clone(),
                })
                .collect();
            CanonicalOperator {
                name: op.name.clone(),
                prevail: op.prevails.clone(),
                pre_post,
                assign_effects,
                cost: op.cost,
            }
        })
        .collect();
}

pub fn filter_unreachable_propositions(task: &mut SASTask) -> Result<usize, SimplifyError> {
    let dtgs = build_dtgs(task);
    let renaming = build_renaming(&dtgs);
    let removed = renaming.num_removed_values;
    renaming.apply_to_task(task)?;
    Ok(removed)
}

pub fn trivial_task(solvable: bool) -> SASTask {
    let variables = vec![Variable {
        value_names: vec!["Atom dummy(val1)".to_string(), "Atom dummy(val2)".to_string()],
    }];
    let ranges = vec![2];
    let init = vec![0];
    let goal = if solvable { vec![(0, 0)] } else { vec![(0, 1)] };
    SASTask {
        variables,
        operators: vec![],
        numeric_variables: vec![],
        numeric_axioms: vec![],
        comparison_axioms: vec![],
        axioms: vec![],
        numeric_init: vec![],
        mutex_groups: vec![],
        ranges: ranges.clone(),
        axiom_layers: vec![-1],
        init,
        goal,
        translation_key: vec![vec!["Atom dummy(val1)".to_string(), "Atom dummy(val2)".to_string()]],
        canonical_variables: vec![CanonicalVariable {
            name: "var0".to_string(),
            axiom_layer: -1,
            values: vec!["Atom dummy(val1)".to_string(), "Atom dummy(val2)".to_string()],
        }],
        canonical_operators: vec![],
        canonical_metric: Some(("<".to_string(), -1)),
        metric: ("<".to_string(), -1),
        global_constraint: Some((0, 0)),
        comp_axiom_layer: 0,
    }
}
