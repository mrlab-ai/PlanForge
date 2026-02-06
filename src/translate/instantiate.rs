use crate::translate::build_model;
use crate::translate::numeric_axiom_rules::InstantiatedNumericAxiom;
use crate::translate::pddl_ast::{Condition, Effect};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct GroundedOp {
    pub name: String,
    pub args: Vec<String>,
    pub pre: Option<Condition>,
    pub eff: Option<Effect>,
    pub effects: Vec<(Vec<Condition>, Effect)>,
}

#[derive(Debug, Clone)]
pub struct GroundedAxiom {
    pub condition: Condition,
    pub effect_atom: String,
}

#[derive(Debug, Clone)]
pub struct ExploreResult {
    pub relaxed_reachable: bool,
    pub model: Vec<build_model::Atom>,
    pub grounded_ops: Vec<GroundedOp>,
    pub grounded_axioms: Vec<GroundedAxiom>,
    pub numeric_axioms: Vec<InstantiatedNumericAxiom>,
    /// Fluent facts - facts that can change during plan execution
    pub fluent_facts: Vec<build_model::Atom>,
    /// Fluent functions - numeric functions that can change
    pub fluent_functions: Vec<String>, // For now, store function names
    /// Initial values for numeric functions: (function_name, args) -> value
    pub init_function_values: HashMap<(String, Vec<String>), f64>,
    /// Constant predicate facts - predicate facts in init that are not fluent
    pub init_constant_predicate_facts: Vec<build_model::Atom>,
    /// Constant numeric facts - numeric function assignments in init that are not fluent
    pub init_constant_numeric_facts: HashMap<(String, Vec<String>), f64>,
    /// Objects grouped by type (type_name -> list of object names)
    pub type_to_objects: HashMap<String, Vec<String>>,
}

#[derive(Debug)]
pub enum InstantiateError {
    EmptyParameterDomain { param: String, typ: String },
    UnsupportedEffect(String),
    NonFluentPredicate(String),
    NonFluentFunction(String),
    FailedSubstitution(String),
    Normalize(String),
}

impl std::fmt::Display for InstantiateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InstantiateError::EmptyParameterDomain { param, typ } => {
                write!(f, "empty domain for parameter {} of type {}", param, typ)
            }
            InstantiateError::UnsupportedEffect(msg) => write!(f, "unsupported effect: {}", msg),
            InstantiateError::NonFluentPredicate(pred) => {
                write!(f, "non-fluent predicate used in effect: {}", pred)
            }
            InstantiateError::NonFluentFunction(func) => {
                write!(f, "non-fluent numeric function used in effect: {}", func)
            }
            InstantiateError::FailedSubstitution(msg) => {
                write!(f, "failed to substitute effect expression: {}", msg)
            }
            InstantiateError::Normalize(msg) => write!(f, "normalize error: {}", msg),
        }
    }
}

impl std::error::Error for InstantiateError {}

#[derive(Debug, Clone)]
struct SymRule {
    conditions: Vec<build_model::SymAtom>,
    effect: build_model::SymAtom,
}

#[derive(Debug, Clone)]
struct RuleWithType {
    rtype: String,
    conditions: Vec<build_model::SymAtom>,
    effect: build_model::SymAtom,
}

fn symatom_key(atom: &build_model::SymAtom) -> String {
    let mut key = atom.predicate.clone();
    key.push('(');
    key.push_str(&atom.args.join(","));
    key.push(')');
    key
}

fn get_variables(atom: &build_model::SymAtom) -> std::collections::HashSet<String> {
    atom.args
        .iter()
        .filter(|a| a.starts_with('?'))
        .cloned()
        .collect()
}

fn get_variables_in_atoms(atoms: &[build_model::SymAtom]) -> std::collections::HashSet<String> {
    let mut result = std::collections::HashSet::new();
    for atom in atoms {
        result.extend(get_variables(atom));
    }
    result
}

fn split_duplicate_arguments(rule: &mut SymRule) {
    fn rename_in_atom(
        atom: &build_model::SymAtom,
        extra_conditions: &mut Vec<build_model::SymAtom>,
    ) -> build_model::SymAtom {
        let mut used: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut new_args = atom.args.clone();
        for (i, arg) in atom.args.iter().enumerate() {
            if arg.starts_with('?') {
                if used.contains(arg) {
                    let new_var_name = format!("{}@{}", arg, extra_conditions.len());
                    new_args[i] = new_var_name.clone();
                    extra_conditions.push(build_model::SymAtom::new(
                        "=".to_string(),
                        vec![arg.clone(), new_var_name],
                    ));
                } else {
                    used.insert(arg.clone());
                }
            }
        }
        build_model::SymAtom::new(atom.predicate.clone(), new_args)
    }

    let mut extra_conditions: Vec<build_model::SymAtom> = Vec::new();
    rule.effect = rename_in_atom(&rule.effect, &mut extra_conditions);
    let mut new_conditions = Vec::new();
    for cond in &rule.conditions {
        new_conditions.push(rename_in_atom(cond, &mut extra_conditions));
    }
    new_conditions.extend(extra_conditions);
    rule.conditions = new_conditions;
}

fn add_object_conditions_to_rules(rules: &mut [SymRule]) -> bool {
    let mut inserted = false;
    for rule in rules.iter_mut() {
        let mut bound_vars: std::collections::HashSet<String> = std::collections::HashSet::new();
        for cond in &rule.conditions {
            for arg in &cond.args {
                if arg.starts_with('?') {
                    bound_vars.insert(arg.clone());
                }
            }
        }
        let mut extra_conditions: Vec<build_model::SymAtom> = Vec::new();
        for arg in &rule.effect.args {
            if !arg.starts_with('?') || bound_vars.contains(arg) {
                continue;
            }
            extra_conditions.push(build_model::SymAtom::new(
                "@object".to_string(),
                vec![arg.clone()],
            ));
            bound_vars.insert(arg.clone());
        }
        if !extra_conditions.is_empty() {
            rule.conditions.extend(extra_conditions);
            inserted = true;
        }
    }
    inserted
}

fn build_type_hierarchy(
    types: &[(String, Option<String>)],
) -> std::collections::HashMap<String, Vec<String>> {
    let mut parent_map: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();
    for (t, parent) in types {
        if let Some(p) = parent {
            parent_map.insert(t.clone(), p.clone());
        }
    }

    let mut hierarchy: std::collections::HashMap<String, Vec<String>> =
        std::collections::HashMap::new();
    for (t, _) in types {
        let mut chain = Vec::new();
        let mut current = t.clone();
        while let Some(parent) = parent_map.get(&current) {
            chain.push(parent.clone());
            if parent == "object" {
                break;
            }
            current = parent.clone();
        }
        hierarchy.insert(t.clone(), chain);
    }
    hierarchy
}

fn convert_trivial_rules_to_facts(rules: &[SymRule]) -> (Vec<SymRule>, Vec<build_model::Atom>) {
    let mut new_rules = Vec::new();
    let mut extra_facts = Vec::new();
    for rule in rules {
        if rule.conditions.is_empty() {
            let has_vars = rule.effect.args.iter().any(|a| a.starts_with('?'));
            if !has_vars {
                extra_facts.push(build_model::Atom {
                    predicate: rule.effect.predicate.clone(),
                    args: rule
                        .effect
                        .args
                        .iter()
                        .map(|s| build_model::Arg::Const(s.clone()))
                        .collect(),
                });
                continue;
            }
        }
        new_rules.push(rule.clone());
    }
    (new_rules, extra_facts)
}

fn get_connected_conditions(conditions: &[build_model::SymAtom]) -> Vec<Vec<build_model::SymAtom>> {
    let mut var_to_conditions: std::collections::HashMap<String, Vec<usize>> =
        std::collections::HashMap::new();
    for (idx, cond) in conditions.iter().enumerate() {
        for var in cond.args.iter().filter(|a| a.starts_with('?')) {
            var_to_conditions.entry(var.clone()).or_default().push(idx);
        }
    }

    let mut adjacency: Vec<std::collections::HashSet<usize>> =
        vec![std::collections::HashSet::new(); conditions.len()];
    for indices in var_to_conditions.values() {
        if indices.len() < 2 {
            continue;
        }
        let first = indices[0];
        for &other in &indices[1..] {
            adjacency[first].insert(other);
            adjacency[other].insert(first);
        }
    }

    let mut visited = vec![false; conditions.len()];
    let mut components: Vec<Vec<build_model::SymAtom>> = Vec::new();
    for i in 0..conditions.len() {
        if visited[i] {
            continue;
        }
        let mut stack = vec![i];
        let mut comp_indices = Vec::new();
        visited[i] = true;
        while let Some(node) = stack.pop() {
            comp_indices.push(node);
            for &nbr in &adjacency[node] {
                if !visited[nbr] {
                    visited[nbr] = true;
                    stack.push(nbr);
                }
            }
        }
        comp_indices.sort();
        let mut comp: Vec<build_model::SymAtom> = comp_indices
            .iter()
            .map(|&idx| conditions[idx].clone())
            .collect();
        comp.sort_by_key(symatom_key);
        components.push(comp);
    }
    components.sort_by(|a, b| symatom_key(&a[0]).cmp(&symatom_key(&b[0])));
    components
}

fn project_rule(
    rule: &SymRule,
    conditions: Vec<build_model::SymAtom>,
    counter: &mut usize,
) -> SymRule {
    let mut effect_vars: Vec<String> = rule
        .effect
        .args
        .iter()
        .filter(|a| a.starts_with('?'))
        .cloned()
        .collect();
    let cond_vars = get_variables_in_atoms(&conditions);
    effect_vars.retain(|v| cond_vars.contains(v));
    effect_vars.sort();
    let predicate = format!("p${}", counter);
    *counter += 1;
    let effect = build_model::SymAtom::new(predicate, effect_vars);
    SymRule { conditions, effect }
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

fn greedy_join(rule: &SymRule, counter: &mut usize) -> Vec<RuleWithType> {
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

fn split_into_binary_rules(rule: &SymRule, counter: &mut usize) -> Vec<RuleWithType> {
    if rule.conditions.len() <= 1 {
        return vec![RuleWithType {
            rtype: "project".to_string(),
            conditions: rule.conditions.clone(),
            effect: rule.effect.clone(),
        }];
    }
    greedy_join(rule, counter)
}

fn split_rule(rule: &SymRule, counter: &mut usize) -> Vec<RuleWithType> {
    let mut important_conditions = Vec::new();
    let mut trivial_conditions = Vec::new();
    for cond in &rule.conditions {
        if cond.args.iter().any(|a| a.starts_with('?')) {
            important_conditions.push(cond.clone());
        } else {
            trivial_conditions.push(cond.clone());
        }
    }

    let components = get_connected_conditions(&important_conditions);
    if components.len() == 1 && trivial_conditions.is_empty() {
        return split_into_binary_rules(rule, counter);
    }

    let mut result = Vec::new();
    let mut projected_rules = Vec::new();
    for conditions in components {
        projected_rules.push(project_rule(rule, conditions, counter));
    }

    for proj in &projected_rules {
        result.extend(split_into_binary_rules(proj, counter));
    }

    let mut combined_conditions: Vec<build_model::SymAtom> =
        projected_rules.iter().map(|r| r.effect.clone()).collect();
    combined_conditions.extend(trivial_conditions);
    let rtype = if combined_conditions.len() >= 2 {
        "product".to_string()
    } else {
        "project".to_string()
    };
    result.push(RuleWithType {
        rtype,
        conditions: combined_conditions,
        effect: rule.effect.clone(),
    });
    result
}

/// High-level exploration step mirroring python/translate/instantiate.py::explore.
///
/// 1. Translate the normalized task into a datalog-style program.
/// 2. Compute its model to discover reachable facts and action instances.
/// 3. Ground operators from model atoms (model-guided, not cartesian product).
/// Explore using a normalized task (preferred).
/// This version builds proper exploration rules from normalized actions.
pub fn explore_normalized(
    norm_task: &crate::translate::normalize::NormalizableTask,
) -> Result<ExploreResult, InstantiateError> {
    eprintln!("DEBUG: explore_normalized() Step 1: build exploration rules");
    // Step 1: Build exploration rules from normalized actions and axioms
    let exploration_rules = crate::translate::normalize::build_exploration_rules(norm_task)
        .map_err(InstantiateError::Normalize)?;
    eprintln!("  Built {} exploration rules", exploration_rules.len());

    let mut prolog_rules: Vec<SymRule> = exploration_rules
        .into_iter()
        .map(|(body, head)| SymRule {
            conditions: body,
            effect: head,
        })
        .collect();

    for rule in prolog_rules.iter_mut() {
        split_duplicate_arguments(rule);
    }

    let object_predicate_required = add_object_conditions_to_rules(&mut prolog_rules);

    let (prolog_rules, extra_facts) = convert_trivial_rules_to_facts(&prolog_rules);

    eprintln!("DEBUG: explore_normalized() Step 1b: split rules");
    let mut split_rules: Vec<RuleWithType> = Vec::new();
    let mut counter = 0;
    for rule in &prolog_rules {
        split_rules.extend(split_rule(rule, &mut counter));
    }
    eprintln!("  After splitting: {} rules", split_rules.len());

    let mut rule_specs: Vec<build_model::RuleSpec> = Vec::new();
    for rule in split_rules {
        rule_specs.push(build_model::RuleSpec {
            rtype: rule.rtype,
            effect: rule.effect,
            conditions: rule.conditions,
        });
    }

    eprintln!("DEBUG: explore_normalized() Step 2: add init facts");
    // Step 2: Build init facts from problem
    let mut init_facts: Vec<build_model::Atom> = Vec::new();
    let type_hierarchy = build_type_hierarchy(&norm_task.types);

    // Add type facts for all objects (direct type and all supertypes)
    for (obj_name, obj_type) in &norm_task.objects {
        let type_name = obj_type.clone().unwrap_or_else(|| "object".to_string());
        init_facts.push(build_model::Atom {
            predicate: type_name.clone(),
            args: vec![build_model::Arg::Const(obj_name.clone())],
        });
        if let Some(supertypes) = type_hierarchy.get(&type_name) {
            for supertype in supertypes {
                init_facts.push(build_model::Atom {
                    predicate: supertype.clone(),
                    args: vec![build_model::Arg::Const(obj_name.clone())],
                });
            }
        } else if type_name != "object" {
            init_facts.push(build_model::Atom {
                predicate: "object".to_string(),
                args: vec![build_model::Arg::Const(obj_name.clone())],
            });
        }
    }
    // Add equality facts for all objects: =(obj, obj)
    for (obj_name, _) in &norm_task.objects {
        init_facts.push(build_model::Atom {
            predicate: "=".to_string(),
            args: vec![
                build_model::Arg::Const(obj_name.clone()),
                build_model::Arg::Const(obj_name.clone()),
            ],
        });
    }
    if object_predicate_required {
        for (obj_name, _) in &norm_task.objects {
            init_facts.push(build_model::Atom {
                predicate: "@object".to_string(),
                args: vec![build_model::Arg::Const(obj_name.clone())],
            });
        }
    }
    // Add init state atoms
    for init_sexpr in &norm_task.init {
        if let Some(atom) = sexpr_to_atom(init_sexpr) {
            init_facts.push(atom);
            continue;
        }

        // Handle numeric function assignments (= (func args) val)
        if let crate::translate::pddl_parser::SExpr::List(items) = init_sexpr {
            if items.len() >= 3 {
                if let crate::translate::pddl_parser::SExpr::Atom(op) = &items[0] {
                    if op == "=" {
                        if let crate::translate::pddl_parser::SExpr::List(func_items) = &items[1] {
                            if !func_items.is_empty() {
                                if let crate::translate::pddl_parser::SExpr::Atom(fname) =
                                    &func_items[0]
                                {
                                    let func_args: Vec<build_model::Arg> = func_items[1..]
                                        .iter()
                                        .filter_map(|item| {
                                            if let crate::translate::pddl_parser::SExpr::Atom(s) =
                                                item
                                            {
                                                Some(build_model::Arg::Const(s.clone()))
                                            } else {
                                                None
                                            }
                                        })
                                        .collect();
                                    init_facts.push(build_model::Atom {
                                        predicate: fname.clone(),
                                        args: func_args.clone(),
                                    });
                                    let defined_pred = format!("defined!{}", fname);
                                    init_facts.push(build_model::Atom {
                                        predicate: defined_pred,
                                        args: func_args,
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Add extra facts from rules with no body
    init_facts.extend(extra_facts);

    eprintln!("  Added {} init facts", init_facts.len());

    eprintln!("DEBUG: explore_normalized() Step 3: compute model");
    // Step 3: Compute the datalog model
    let mut rules = build_model::convert_rules(&rule_specs);
    let model = build_model::compute_model(&mut rules, &init_facts);

    eprintln!("DEBUG: computed model with {} atoms", model.len());

    eprintln!("DEBUG: explore_normalized() Step 4: extract fluent facts and functions");
    // Step 4: Extract fluent facts and functions from model
    let fluent_facts = get_fluent_facts(norm_task, &model);
    let fluent_functions = get_fluent_functions(norm_task);
    eprintln!(
        "  Fluent facts: {}, fluent functions: {}",
        fluent_facts.len(),
        fluent_functions.len()
    );

    let fluent_predicates = crate::translate::normalize::get_fluent_predicates(norm_task);
    let init_atom_set = build_init_atom_set(&init_facts);
    let model_atom_set = build_model_atom_set(&model);
    let init_function_values = extract_init_function_values(norm_task);
    let type_to_objects = get_objects_by_type(&norm_task.objects, &type_hierarchy);

    eprintln!("DEBUG: explore_normalized() Step 5: ground actions from model");
    // Step 5: Extract grounded actions from model
    let (ops, num_axioms, grounded_axioms) = ground_from_normalized_model(
        &model,
        norm_task,
        &init_atom_set,
        &model_atom_set,
        &fluent_predicates,
        &type_to_objects,
        &fluent_functions,
        &init_function_values,
    )?;
    eprintln!("DEBUG: grounded {} operators", ops.len());

    let relaxed_reachable = model.iter().any(|atom| atom.predicate == "@goal-reachable");

    eprintln!("DEBUG: explore_normalized() Step 6: separate init state into constants and fluents");
    // Step 6: Extract init function values and separate constant facts
    let init_constant_numeric_facts =
        extract_constant_numeric_facts(&init_function_values, &fluent_functions);
    let init_constant_predicate_facts =
        extract_constant_predicate_facts(&init_facts, &fluent_facts);
    eprintln!(
        "  Init function values: {}, constant numeric: {}, constant predicates: {}",
        init_function_values.len(),
        init_constant_numeric_facts.len(),
        init_constant_predicate_facts.len()
    );
    eprintln!("  Type-to-objects mapping: {} types", type_to_objects.len());

    Ok(ExploreResult {
        relaxed_reachable,
        model,
        grounded_ops: ops,
        grounded_axioms,
        numeric_axioms: num_axioms,
        fluent_facts,
        fluent_functions,
        init_function_values,
        init_constant_predicate_facts,
        init_constant_numeric_facts,
        type_to_objects,
    })
}

/// Extract fluent facts from model based on fluent predicates.
/// Fluent facts are facts whose predicate can change during plan execution.
fn get_fluent_facts(
    norm_task: &crate::translate::normalize::NormalizableTask,
    model: &[build_model::Atom],
) -> Vec<build_model::Atom> {
    use crate::translate::normalize;

    let fluent_predicates = normalize::get_fluent_predicates(norm_task);

    model
        .iter()
        .filter(|atom| fluent_predicates.contains(&atom.predicate))
        .cloned()
        .collect()
}

/// Extract fluent functions (numeric functions) from normalized task.
/// These are numeric functions that can change during plan execution.
/// A numeric function is fluent if it appears in any action effect.
fn get_fluent_functions(norm_task: &crate::translate::normalize::NormalizableTask) -> Vec<String> {
    use crate::translate::pddl_parser::SExpr;
    use std::collections::HashSet;

    let mut fluent_functions = HashSet::new();

    for action in &norm_task.actions {
        for effect in &action.effects {
            if let SExpr::List(items) = &effect.effect {
                if let Some(SExpr::Atom(op)) = items.get(0) {
                    if matches!(
                        op.as_str(),
                        "increase" | "decrease" | "assign" | "scale-up" | "scale-down"
                    ) {
                        if let Some(SExpr::List(func_items)) = items.get(1) {
                            if let Some(SExpr::Atom(func_name)) = func_items.get(0) {
                                fluent_functions.insert(func_name.clone());
                            }
                        }
                    }
                }
            }
        }
    }

    fluent_functions.into_iter().collect()
}

fn build_init_atom_set(
    init_facts: &[build_model::Atom],
) -> std::collections::HashSet<(String, Vec<String>)> {
    init_facts
        .iter()
        .map(|atom| {
            let args = atom
                .args
                .iter()
                .filter_map(|arg| match arg {
                    build_model::Arg::Const(s) => Some(s.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>();
            (atom.predicate.clone(), args)
        })
        .collect()
}

fn build_model_atom_set(
    model: &[build_model::Atom],
) -> std::collections::HashSet<(String, Vec<String>)> {
    model
        .iter()
        .map(|atom| {
            let args = atom
                .args
                .iter()
                .filter_map(|arg| match arg {
                    build_model::Arg::Const(s) => Some(s.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>();
            (atom.predicate.clone(), args)
        })
        .collect()
}

/// Extract function names from an effect SExpr.
/// Handles (increase (func args) value), (decrease ...), (assign ...), etc.
#[allow(dead_code)]
fn extract_function_names_from_effect(
    effect: &crate::translate::pddl_parser::SExpr,
    result: &mut std::collections::HashSet<String>,
) {
    use crate::translate::pddl_parser::SExpr;

    if let SExpr::List(items) = effect {
        if items.len() >= 2 {
            // Check for numeric effect operators
            if let SExpr::Atom(op) = &items[0] {
                if matches!(
                    op.as_str(),
                    "increase" | "decrease" | "assign" | "scale-up" | "scale-down"
                ) {
                    // Second element should be the function call: (func-name args...)
                    if let SExpr::List(func_call) = &items[1] {
                        if !func_call.is_empty() {
                            if let SExpr::Atom(func_name) = &func_call[0] {
                                result.insert(func_name.clone());
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Check if a predicate represents a numeric function (kept for future use).
#[allow(dead_code)]
fn is_numeric_function(predicate: &str) -> bool {
    // Common numeric effect predicates
    matches!(
        predicate,
        "increase" | "decrease" | "assign" | "scale-up" | "scale-down"
    ) || predicate.starts_with("f#") // derived function prefix (if used)
}

/// Extract initial values for all numeric functions from init facts.
/// Returns a map from (function_name, args) to initial value.
///
/// Parses init SExprs looking for numeric assignments like: (= (function-name obj1 obj2) value)
fn extract_init_function_values(
    norm_task: &crate::translate::normalize::NormalizableTask,
) -> HashMap<(String, Vec<String>), f64> {
    use crate::translate::function_expression::FunctionalExpression;
    use crate::translate::pddl_parser::SExpr;

    let mut init_values = HashMap::new();

    for init_sexpr in &norm_task.init {
        // Look for (= (function-name args...) value) patterns
        if let SExpr::List(items) = init_sexpr {
            if items.len() == 3 {
                // Check if it's an assignment: first element is "="
                if let SExpr::Atom(op) = &items[0] {
                    if op == "=" {
                        // Second element should be a function call: (function-name args...)
                        if let SExpr::List(func_call) = &items[1] {
                            if !func_call.is_empty() {
                                if let SExpr::Atom(func_name) = &func_call[0] {
                                    // Extract arguments
                                    let mut args = Vec::new();
                                    for arg in &func_call[1..] {
                                        if let SExpr::Atom(arg_name) = arg {
                                            args.push(arg_name.clone());
                                        }
                                    }

                                    // Third element is the numeric value
                                    if let SExpr::Atom(value_str) = &items[2] {
                                        if let Ok(value) = value_str.parse::<f64>() {
                                            init_values.insert((func_name.clone(), args), value);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    for axiom in &norm_task.numeric_axioms {
        if axiom.op.is_none() && axiom.parameters.is_empty() && axiom.parts.len() == 1 {
            if let FunctionalExpression::Constant(constant) = &axiom.parts[0] {
                init_values.insert((axiom.name.clone(), vec![]), constant.value as f64);
            }
        }
    }

    init_values
}

/// Extract constant numeric facts - numeric functions in init that are not fluent.
/// These are numeric functions whose values never change during plan execution.
fn extract_constant_numeric_facts(
    init_function_values: &HashMap<(String, Vec<String>), f64>,
    fluent_functions: &[String],
) -> HashMap<(String, Vec<String>), f64> {
    let fluent_set: std::collections::HashSet<_> = fluent_functions.iter().collect();

    init_function_values
        .iter()
        .filter(|((func_name, _args), _value)| !fluent_set.contains(func_name))
        .map(|(k, v)| (k.clone(), *v))
        .collect()
}

/// Extract constant predicate facts - predicate facts in init that are not fluent.
/// These are facts that never change during plan execution.
fn extract_constant_predicate_facts(
    init_facts: &[build_model::Atom],
    fluent_facts: &[build_model::Atom],
) -> Vec<build_model::Atom> {
    // Convert fluent_facts to a set for efficient lookup
    let fluent_set: std::collections::HashSet<_> = fluent_facts.iter().collect();

    init_facts
        .iter()
        .filter(|atom| {
            // Only consider regular predicate atoms (not type facts, not "=" intermediate facts)
            !fluent_set.contains(atom) && atom.predicate != "="
        })
        .cloned()
        .collect()
}

/// Get objects grouped by type.
/// Returns a map from type name to list of object names of that type.
///
/// Implements type hierarchy support: each object appears under its direct type
/// and all supertypes. For PDDL domains, we assume all types inherit from "object"
/// as this is the standard PDDL convention.
///
/// Based on Python's instantiate.py:get_objects_by_type()
fn get_objects_by_type(
    objects: &[(String, Option<String>)],
    type_hierarchy: &std::collections::HashMap<String, Vec<String>>,
) -> HashMap<String, Vec<String>> {
    let mut result: HashMap<String, Vec<String>> = HashMap::new();

    for (obj_name, obj_type) in objects {
        let type_name = obj_type.clone().unwrap_or_else(|| "object".to_string());
        result
            .entry(type_name.clone())
            .or_insert_with(Vec::new)
            .push(obj_name.clone());

        if let Some(supertypes) = type_hierarchy.get(&type_name) {
            for supertype in supertypes {
                result
                    .entry(supertype.clone())
                    .or_insert_with(Vec::new)
                    .push(obj_name.clone());
            }
        } else if type_name != "object" {
            result
                .entry("object".to_string())
                .or_insert_with(Vec::new)
                .push(obj_name.clone());
        }
    }

    result
}

/// Ground actions from model using normalized task actions.
fn ground_from_normalized_model(
    model: &[build_model::Atom],
    norm_task: &crate::translate::normalize::NormalizableTask,
    init_atom_set: &std::collections::HashSet<(String, Vec<String>)>,
    model_atom_set: &std::collections::HashSet<(String, Vec<String>)>,
    fluent_predicates: &std::collections::HashSet<String>,
    type_to_objects: &std::collections::HashMap<String, Vec<String>>,
    fluent_functions: &[String],
    init_function_values: &std::collections::HashMap<(String, Vec<String>), f64>,
) -> Result<(Vec<GroundedOp>, Vec<InstantiatedNumericAxiom>, Vec<GroundedAxiom>), InstantiateError> {
    use std::collections::{HashMap, HashSet};

    // Build action lookup map
    let mut action_map: HashMap<String, &crate::translate::normalize::TaskAction> = HashMap::new();
    for action in &norm_task.actions {
        action_map.insert(action.name.clone(), action);
    }

    // Build propositional axiom lookup map by name
    let mut axiom_map: HashMap<String, &crate::translate::normalize::TaskAxiom> = HashMap::new();
    for axiom in &norm_task.axioms {
        axiom_map.insert(axiom.name.clone(), axiom);
    }

    // Build numeric axiom lookup map by name
    let mut numeric_axiom_map: HashMap<
        String,
        &crate::translate::normalization_function_admin::NumericAxiom,
    > = HashMap::new();
    for axiom in &norm_task.numeric_axioms {
        numeric_axiom_map.insert(axiom.name.clone(), axiom);
    }

    eprintln!(
        "DEBUG: numeric_axiom_map keys: {:?}",
        numeric_axiom_map.keys().collect::<Vec<_>>()
    );

    let mut grounded_ops = Vec::new();
    let mut grounded_axioms: Vec<GroundedAxiom> = Vec::new();
    let mut grounded_axiom_atoms: HashSet<String> = HashSet::new();
    let mut instantiated_numeric_axioms: HashSet<InstantiatedNumericAxiom> = HashSet::new();

    // Track numeric axiom atoms found
    let mut numeric_axiom_atom_count = 0;

    // First pass: iterate model atoms and extract action instantiations
    for atom in model {
        // Check if this atom represents an action (predicate starts with @action-)
        if atom.predicate.starts_with("@action-") {
            let action_name = &atom.predicate["@action-".len()..];

            if let Some(action) = action_map.get(action_name) {
                // Extract grounded arguments from atom
                let grounded_args = extract_grounded_args(&atom.args);

                // Create variable mapping: parameter name -> grounded object
                let variable_mapping = create_variable_mapping(&action.parameters, &grounded_args);

                if action.name == "move_up" || action.name == "load" {
                    eprintln!(
                        "DEBUG: action {} effects={} precondition={:?}",
                        action.name,
                        action.effects.len(),
                        action.precondition
                    );
                    for (idx, eff) in action.effects.iter().enumerate() {
                        eprintln!("DEBUG: action {} effect[{}]={:?}", action.name, idx, eff);
                    }
                }

                // Instantiate this specific action with these parameters
                let grounded_op = instantiate_normalized_action(
                    action,
                    &grounded_args,
                    &variable_mapping,
                    init_atom_set,
                    model_atom_set,
                    fluent_predicates,
                    type_to_objects,
                    fluent_functions,
                    init_function_values,
                )?;
                if let Some(op) = grounded_op {
                    grounded_ops.push(op);
                }
            }
        }
    }

    // Build set of fluent function instances from grounded actions
    let mut fluent_instances: HashSet<(String, Vec<String>)> = HashSet::new();
    for op in &grounded_ops {
        for (_conds, eff) in &op.effects {
            match eff {
                Effect::Increase(name, args, _) | Effect::Decrease(name, args, _) => {
                    fluent_instances.insert((name.clone(), args.clone()));
                }
                Effect::And(v) => {
                    for sub in v {
                        if let Effect::Increase(name, args, _) | Effect::Decrease(name, args, _) =
                            sub
                        {
                            fluent_instances.insert((name.clone(), args.clone()));
                        }
                    }
                }
                _ => {}
            }
        }
        if let Some(eff) = &op.eff {
            match eff {
                Effect::Increase(name, args, _) | Effect::Decrease(name, args, _) => {
                    fluent_instances.insert((name.clone(), args.clone()));
                }
                Effect::And(v) => {
                    for sub in v {
                        if let Effect::Increase(name, args, _) | Effect::Decrease(name, args, _) =
                            sub
                        {
                            fluent_instances.insert((name.clone(), args.clone()));
                        }
                    }
                }
                _ => {}
            }
        }
    }

    // Second pass: instantiate numeric axioms with fluent-instance knowledge
    for atom in model {
        if let Some(axiom) = numeric_axiom_map.get(&atom.predicate) {
            numeric_axiom_atom_count += 1;
            if numeric_axiom_atom_count <= 5 {
                eprintln!(
                    "DEBUG: Found numeric axiom atom: {} args={:?}",
                    atom.predicate, atom.args
                );
            }
            // Extract grounded arguments
            let grounded_args = extract_grounded_args(&atom.args);
            if grounded_args.len() < axiom.parameters.len() {
                eprintln!(
                    "DEBUG: Skipping numeric axiom {} due to arg mismatch: params={}, args={}",
                    axiom.name,
                    axiom.parameters.len(),
                    grounded_args.len()
                );
                continue;
            }

            // Build variable mapping from axiom parameters to grounded args
            let variable_mapping: HashMap<String, String> = axiom
                .parameters
                .iter()
                .zip(grounded_args.iter().take(axiom.parameters.len()))
                .map(|(param, obj)| (param.clone(), obj.clone()))
                .collect();

            // Instantiate the numeric axiom
            if let Some(inst_axiom) = instantiate_numeric_axiom(
                axiom,
                &variable_mapping,
                &fluent_instances,
                init_function_values,
            ) {
                instantiated_numeric_axioms.insert(inst_axiom);
            } else {
                eprintln!(
                    "DEBUG: instantiate_numeric_axiom returned None for {} with mapping {:?}",
                    axiom.name, variable_mapping
                );
            }
        }
    }

    // Third pass: instantiate propositional axioms from the model
    for atom in model {
        if let Some(axiom) = axiom_map.get(&atom.predicate) {
            let grounded_args = extract_grounded_args(&atom.args);
            if grounded_args.len() < axiom.parameters.len() {
                eprintln!(
                    "DEBUG: Skipping axiom {} due to arg mismatch: params={}, args={}",
                    axiom.name,
                    axiom.parameters.len(),
                    grounded_args.len()
                );
                continue;
            }

            let used_args: Vec<String> = grounded_args
                .iter()
                .take(axiom.parameters.len())
                .cloned()
                .collect();
            let variable_mapping = create_variable_mapping(&axiom.parameters, &used_args);
            let condition = crate::translate::pddl_ast::substitute_condition(
                &axiom.condition,
                &variable_mapping,
            );
            let effect_atom = format!("{}({})", axiom.name, used_args.join(", "));
            if grounded_axiom_atoms.insert(effect_atom.clone()) {
                grounded_axioms.push(GroundedAxiom {
                    condition,
                    effect_atom,
                });
            }
        }
    }

    eprintln!(
        "DEBUG: total numeric axiom atoms found: {}, instantiated axioms: {}",
        numeric_axiom_atom_count,
        instantiated_numeric_axioms.len()
    );

    let num_axioms: Vec<InstantiatedNumericAxiom> =
        instantiated_numeric_axioms.into_iter().collect();

    Ok((grounded_ops, num_axioms, grounded_axioms))
}

/// Instantiate a numeric axiom with specific variable bindings.
fn instantiate_numeric_axiom(
    axiom: &crate::translate::normalization_function_admin::NumericAxiom,
    variable_mapping: &std::collections::HashMap<String, String>,
    fluent_instances: &std::collections::HashSet<(String, Vec<String>)>,
    init_function_values: &std::collections::HashMap<(String, Vec<String>), f64>,
) -> Option<InstantiatedNumericAxiom> {
    use crate::translate::function_expression::FunctionalExpression;
    use crate::translate::numeric_axiom_rules::{
        NumericConstant, NumericPart, PrimitiveNumericExpression,
    };

    // Build instantiated name: "(axiom-name arg1 arg2 ...)"
    let arg_list: Vec<String> = axiom
        .parameters
        .iter()
        .map(|p| {
            variable_mapping
                .get(p)
                .cloned()
                .unwrap_or_else(|| p.clone())
        })
        .collect();
    let inst_name = format!("({} {})", axiom.name, arg_list.join(" "));

    // Instantiate each part
    let mut inst_parts = Vec::new();
    for part in &axiom.parts {
        match part {
            FunctionalExpression::Constant(nc) => {
                inst_parts.push(NumericPart::Constant(NumericConstant(nc.value)));
            }
            FunctionalExpression::Primitive(pne) => {
                // Substitute variables in args
                let inst_args: Vec<String> = pne
                    .args
                    .iter()
                    .map(|a| {
                        variable_mapping
                            .get(a)
                            .cloned()
                            .unwrap_or_else(|| a.clone())
                    })
                    .collect();
                let inst_key = (pne.symbol.clone(), inst_args.clone());
                if !fluent_instances.contains(&inst_key) {
                    if let Some(val) = init_function_values.get(&inst_key) {
                        let iv = *val as i64;
                        inst_parts.push(NumericPart::Constant(NumericConstant(iv)));
                        continue;
                    }
                }
                let inst_pne = PrimitiveNumericExpression {
                    name: pne.symbol.clone(),
                    args: inst_args,
                };
                inst_parts.push(NumericPart::Primitive(inst_pne));
            }
            _ => {
                // For nested arithmetic expressions, we'd need recursive instantiation
                // For now, skip (normalization should have flattened these)
            }
        }
    }

    // Build effect PNE
    let effect_args: Vec<String> = axiom
        .parameters
        .iter()
        .map(|p| {
            variable_mapping
                .get(p)
                .cloned()
                .unwrap_or_else(|| p.clone())
        })
        .collect();
    let effect = PrimitiveNumericExpression {
        name: axiom.name.clone(),
        args: effect_args,
    };

    Some(InstantiatedNumericAxiom {
        name: inst_name,
        op: axiom.op.clone(),
        parts: inst_parts,
        effect,
    })
}

/// Extract grounded (constant) arguments from model atom args.
fn extract_grounded_args(args: &[build_model::Arg]) -> Vec<String> {
    args.iter()
        .map(|arg| match arg {
            build_model::Arg::Const(s) => s.clone(),
            build_model::Arg::FreeVar(s) => s.clone(),
            build_model::Arg::Var(_) => {
                // Model should only contain constants after grounding
                eprintln!("Warning: Found Var in model atom, treating as placeholder");
                "?unknown".to_string()
            }
        })
        .collect()
}

/// Create variable mapping from action parameters to grounded arguments.
fn create_variable_mapping(
    parameters: &[(String, Option<String>)],
    grounded_args: &[String],
) -> std::collections::HashMap<String, String> {
    parameters
        .iter()
        .zip(grounded_args.iter())
        .map(|((param_name, _type), obj)| {
            // Remove '?' prefix from parameter if present
            let clean_param = if param_name.starts_with('?') {
                param_name[1..].to_string()
            } else {
                param_name.clone()
            };
            (format!("?{}", clean_param), obj.clone())
        })
        .collect()
}

/// Instantiate a normalized action with specific grounded arguments.
fn instantiate_normalized_action(
    action: &crate::translate::normalize::TaskAction,
    grounded_args: &[String],
    variable_mapping: &std::collections::HashMap<String, String>,
    init_atom_set: &std::collections::HashSet<(String, Vec<String>)>,
    model_atom_set: &std::collections::HashSet<(String, Vec<String>)>,
    fluent_predicates: &std::collections::HashSet<String>,
    type_to_objects: &std::collections::HashMap<String, Vec<String>>,
    fluent_functions: &[String],
    init_function_values: &std::collections::HashMap<(String, Vec<String>), f64>,
) -> Result<Option<GroundedOp>, InstantiateError> {
    use crate::translate::pddl_ast;

    let name = format!("{}({})", action.name, grounded_args.join(","));

    // Substitute variables in precondition
    let pre_sub = pddl_ast::substitute_condition(&action.precondition, variable_mapping);
    let fluent_function_set: std::collections::HashSet<String> =
        fluent_functions.iter().cloned().collect();
    let _preconditions = match instantiate_condition_list(
        &pre_sub,
        init_atom_set,
        model_atom_set,
        fluent_predicates,
        &fluent_function_set,
        init_function_values,
    ) {
        Some(list) => list,
        None => {
            eprintln!(
                "DEBUG: skipping action {} due to preconditions {:?}",
                name, pre_sub
            );
            return Ok(None);
        }
    };

    let effects = instantiate_effects(
        action,
        variable_mapping,
        init_atom_set,
        model_atom_set,
        fluent_predicates,
        type_to_objects,
        fluent_functions,
        init_function_values,
    )?;

    if effects.is_empty() {
        eprintln!("DEBUG: skipping action {} due to empty effects", name);
        return Ok(None);
    }

    let eff_sub = if effects.len() == 1 && effects[0].0.is_empty() {
        Some(effects[0].1.clone())
    } else if effects.is_empty() {
        None
    } else {
        Some(pddl_ast::Effect::And(
            effects.iter().map(|(_, e)| e.clone()).collect(),
        ))
    };

    Ok(Some(GroundedOp {
        name,
        args: grounded_args.to_vec(),
        pre: Some(pre_sub),
        eff: eff_sub,
        effects,
    }))
}

fn instantiate_effects(
    action: &crate::translate::normalize::TaskAction,
    base_mapping: &std::collections::HashMap<String, String>,
    init_atom_set: &std::collections::HashSet<(String, Vec<String>)>,
    model_atom_set: &std::collections::HashSet<(String, Vec<String>)>,
    fluent_predicates: &std::collections::HashSet<String>,
    type_to_objects: &std::collections::HashMap<String, Vec<String>>,
    fluent_functions: &[String],
    init_function_values: &std::collections::HashMap<(String, Vec<String>), f64>,
) -> Result<Vec<(Vec<Condition>, Effect)>, InstantiateError> {
    let mut effects = Vec::new();
    for effect in &action.effects {
        instantiate_effect_with_params(
            effect,
            base_mapping,
            init_atom_set,
            model_atom_set,
            fluent_predicates,
            type_to_objects,
            fluent_functions,
            init_function_values,
            &mut effects,
        )?;
    }

    Ok(effects)
}

fn instantiate_effect_with_params(
    effect: &crate::translate::normalize::TaskEffect,
    base_mapping: &std::collections::HashMap<String, String>,
    init_atom_set: &std::collections::HashSet<(String, Vec<String>)>,
    model_atom_set: &std::collections::HashSet<(String, Vec<String>)>,
    fluent_predicates: &std::collections::HashSet<String>,
    type_to_objects: &std::collections::HashMap<String, Vec<String>>,
    fluent_functions: &[String],
    init_function_values: &std::collections::HashMap<(String, Vec<String>), f64>,
    out: &mut Vec<(Vec<Condition>, Effect)>,
) -> Result<(), InstantiateError> {
    let parameter_lists = effect
        .parameters
        .iter()
        .map(|(name, typ)| {
            let key = typ.clone().unwrap_or_else(|| "object".to_string());
            let values = type_to_objects.get(&key).cloned().unwrap_or_default();
            if values.is_empty() {
                return Err(InstantiateError::EmptyParameterDomain {
                    param: name.clone(),
                    typ: key,
                });
            }
            Ok((name.clone(), values))
        })
        .collect::<Result<Vec<_>, InstantiateError>>()?;

    let assignments = cartesian_assignments(&parameter_lists);
    let fluent_function_set: std::collections::HashSet<String> =
        fluent_functions.iter().cloned().collect();

    for assignment in assignments {
        let mut mapping = base_mapping.clone();
        for (param, value) in assignment {
            mapping.insert(param, value);
        }

        let condition =
            crate::translate::pddl_ast::substitute_condition(&effect.condition, &mapping);
        let cond_list = match instantiate_condition_list(
            &condition,
            init_atom_set,
            model_atom_set,
            fluent_predicates,
            &fluent_function_set,
            init_function_values,
        ) {
            Some(list) => list,
            None => continue,
        };

        let substituted = match substitute_sexpr_with_numeric(
            &effect.effect,
            &mapping,
            &fluent_function_set,
            init_function_values,
        ) {
            Some(expr) => expr,
            None => {
                return Err(InstantiateError::FailedSubstitution(format!(
                    "{:?}",
                    effect.effect
                )))
            }
        };
        let parsed = crate::translate::pddl_ast::sexpr_to_effect(&substituted);
        match parsed {
            crate::translate::pddl_ast::Effect::Add(name, args) => {
                if fluent_predicates.contains(&name) {
                    out.push((
                        cond_list,
                        crate::translate::pddl_ast::Effect::Add(name, args),
                    ));
                } else if !init_atom_set.contains(&(name.clone(), args.clone())) {
                    return Err(InstantiateError::NonFluentPredicate(name));
                }
            }
            crate::translate::pddl_ast::Effect::Del(name, args) => {
                if fluent_predicates.contains(&name) {
                    out.push((
                        cond_list,
                        crate::translate::pddl_ast::Effect::Del(name, args),
                    ));
                } else if init_atom_set.contains(&(name.clone(), args.clone())) {
                    return Err(InstantiateError::NonFluentPredicate(name));
                }
            }
            crate::translate::pddl_ast::Effect::Increase(name, args, val) => {
                if fluent_function_set.contains(&name) {
                    out.push((
                        cond_list,
                        crate::translate::pddl_ast::Effect::Increase(name, args, val),
                    ));
                } else {
                    return Err(InstantiateError::NonFluentFunction(name));
                }
            }
            crate::translate::pddl_ast::Effect::Decrease(name, args, val) => {
                if fluent_function_set.contains(&name) {
                    out.push((
                        cond_list,
                        crate::translate::pddl_ast::Effect::Decrease(name, args, val),
                    ));
                } else {
                    return Err(InstantiateError::NonFluentFunction(name));
                }
            }
            crate::translate::pddl_ast::Effect::And(v) => {
                for sub in v {
                    out.push((cond_list.clone(), sub));
                }
            }
        }
    }
    Ok(())
}

fn cartesian_assignments(parameter_lists: &[(String, Vec<String>)]) -> Vec<Vec<(String, String)>> {
    if parameter_lists.is_empty() {
        return vec![Vec::new()];
    }
    let mut results: Vec<Vec<(String, String)>> = vec![Vec::new()];
    for (name, values) in parameter_lists {
        if values.is_empty() {
            return Vec::new();
        }
        let mut next = Vec::new();
        for prefix in &results {
            for value in values {
                let mut new_prefix = prefix.clone();
                new_prefix.push((name.clone(), value.clone()));
                next.push(new_prefix);
            }
        }
        results = next;
    }
    results
}

fn substitute_sexpr_with_numeric(
    sexpr: &crate::translate::pddl_parser::SExpr,
    mapping: &std::collections::HashMap<String, String>,
    fluent_function_set: &std::collections::HashSet<String>,
    init_function_values: &std::collections::HashMap<(String, Vec<String>), f64>,
) -> Option<crate::translate::pddl_parser::SExpr> {
    use crate::translate::pddl_parser::SExpr;

    match sexpr {
        SExpr::Atom(a) => {
            if a.starts_with('?') {
                if let Some(v) = mapping.get(a) {
                    Some(SExpr::Atom(v.clone()))
                } else {
                    Some(SExpr::Atom(a.clone()))
                }
            } else {
                if a.parse::<f64>().is_ok() {
                    Some(SExpr::Atom(a.clone()))
                } else if fluent_function_set.contains(a) {
                    Some(SExpr::List(vec![SExpr::Atom(a.clone())]))
                } else if let Some(val) = init_function_values.get(&(a.clone(), vec![])) {
                    Some(SExpr::Atom(format_number(*val)))
                } else {
                    Some(SExpr::Atom(a.clone()))
                }
            }
        }
        SExpr::List(list) => {
            if list.is_empty() {
                return Some(SExpr::List(vec![]));
            }
            let op = match &list[0] {
                SExpr::Atom(a) => a.as_str(),
                _ => "",
            };
            if matches!(op, "+" | "-" | "*" | "/") {
                let mut new_items = Vec::new();
                for item in list {
                    new_items.push(substitute_sexpr_with_numeric(
                        item,
                        mapping,
                        fluent_function_set,
                        init_function_values,
                    )?);
                }
                return Some(SExpr::List(new_items));
            }
            if let SExpr::Atom(fname) = &list[0] {
                let mut args = Vec::new();
                for item in &list[1..] {
                    if let SExpr::Atom(arg) = item {
                        if arg.starts_with('?') {
                            args.push(mapping.get(arg).cloned().unwrap_or_else(|| arg.clone()));
                        } else {
                            args.push(arg.clone());
                        }
                    }
                }
                if fluent_function_set.contains(fname) {
                    let mut items = vec![SExpr::Atom(fname.clone())];
                    items.extend(args.into_iter().map(SExpr::Atom));
                    return Some(SExpr::List(items));
                }
                if let Some(val) = init_function_values.get(&(fname.clone(), args.clone())) {
                    return Some(SExpr::Atom(format_number(*val)));
                }
            }
            let mut new_items = Vec::new();
            for item in list {
                new_items.push(substitute_sexpr_with_numeric(
                    item,
                    mapping,
                    fluent_function_set,
                    init_function_values,
                )?);
            }
            Some(SExpr::List(new_items))
        }
    }
}

fn instantiate_condition_list(
    condition: &crate::translate::pddl_ast::Condition,
    init_atom_set: &std::collections::HashSet<(String, Vec<String>)>,
    model_atom_set: &std::collections::HashSet<(String, Vec<String>)>,
    fluent_predicates: &std::collections::HashSet<String>,
    fluent_function_set: &std::collections::HashSet<String>,
    init_function_values: &std::collections::HashMap<(String, Vec<String>), f64>,
) -> Option<Vec<Condition>> {
    use crate::translate::pddl_ast::Condition;

    match condition {
        Condition::True => Some(Vec::new()),
        Condition::Atom(pred, args) => {
            if fluent_predicates.contains(pred) {
                Some(vec![Condition::Atom(pred.clone(), args.clone())])
            } else if init_atom_set.contains(&(pred.clone(), args.clone())) {
                Some(Vec::new())
            } else if model_atom_set.contains(&(pred.clone(), args.clone())) {
                Some(Vec::new())
            } else {
                None
            }
        }
        Condition::Not(inner) => match inner.as_ref() {
            Condition::Atom(pred, args) => {
                if fluent_predicates.contains(pred) {
                    Some(vec![Condition::Not(Box::new(Condition::Atom(
                        pred.clone(),
                        args.clone(),
                    )))])
                } else if init_atom_set.contains(&(pred.clone(), args.clone()))
                    || model_atom_set.contains(&(pred.clone(), args.clone()))
                {
                    None
                } else {
                    Some(Vec::new())
                }
            }
            _ => Some(vec![condition.clone()]),
        },
        Condition::And(parts) => {
            let mut result = Vec::new();
            for part in parts {
                let mut part_list = instantiate_condition_list(
                    part,
                    init_atom_set,
                    model_atom_set,
                    fluent_predicates,
                    fluent_function_set,
                    init_function_values,
                )?;
                result.append(&mut part_list);
            }
            Some(result)
        }
        Condition::Comparison(op, left, right) => {
            let left_sub = substitute_sexpr_with_numeric(
                left,
                &std::collections::HashMap::new(),
                fluent_function_set,
                init_function_values,
            )?;
            let right_sub = substitute_sexpr_with_numeric(
                right,
                &std::collections::HashMap::new(),
                fluent_function_set,
                init_function_values,
            )?;
            Some(vec![Condition::Comparison(op.clone(), left_sub, right_sub)])
        }
        _ => Some(vec![condition.clone()]),
    }
}

fn format_number(value: f64) -> String {
    if value.fract() == 0.0 {
        format!("{}", value as i64)
    } else {
        value.to_string()
    }
}

/// Convert a PDDL SExpr to a build_model Atom.
fn sexpr_to_atom(sexpr: &crate::translate::pddl_parser::SExpr) -> Option<build_model::Atom> {
    use crate::translate::pddl_parser::SExpr;
    match sexpr {
        SExpr::List(items) if !items.is_empty() => {
            if let SExpr::Atom(pred) = &items[0] {
                if pred == "=" {
                    return None;
                }
                let args: Vec<build_model::Arg> = items[1..]
                    .iter()
                    .filter_map(|item| {
                        if let SExpr::Atom(s) = item {
                            Some(build_model::Arg::Const(s.clone()))
                        } else {
                            None
                        }
                    })
                    .collect();
                Some(build_model::Atom {
                    predicate: pred.clone(),
                    args,
                })
            } else {
                None
            }
        }
        _ => None,
    }
}
