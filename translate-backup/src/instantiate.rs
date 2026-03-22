#[cfg(test)]
mod tests;

use crate::translate::build_model;
use crate::translate::function_expression::format_float;
use crate::translate::numeric_axiom_rules::InstantiatedNumericAxiom;
use crate::translate::pddl::{Condition, Effect};
use crate::translate::pddl_parser::SExpr;
use crate::translate::pddl_to_prolog;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct GroundedOp {
    pub name: String,
    pub args: Vec<String>,
    pub pre: Option<Condition>,
    pub eff: Option<Effect>,
    pub effects: Vec<(Vec<Condition>, Effect)>,
    pub cost: f64,
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
    /// Stored as fully-qualified PNE strings like "f(a,b)" or "g()".
    pub fluent_functions: Vec<String>,
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
    eprintln!("DEBUG: explore_normalized() Step 1: build prolog program");
    let prog = pddl_to_prolog::translate(norm_task);
    let type_hierarchy = build_type_hierarchy(&norm_task.types);

    eprintln!(
        "  Prolog facts: {}, rules: {}",
        prog.facts.len(),
        prog.rules.len()
    );

    let init_facts: Vec<build_model::Atom> = prog.facts.iter().map(|f| f.atom.clone()).collect();

    let mut rule_specs: Vec<build_model::RuleSpec> = Vec::new();
    for rule in &prog.rules {
        rule_specs.push(build_model::RuleSpec {
            rtype: rule.rtype.clone(),
            effect: rule.effect.clone(),
            conditions: rule.conditions.clone(),
        });
    }

    eprintln!("DEBUG: explore_normalized() Step 2: compute model");
    // Step 3: Compute the datalog model
    let mut rules = build_model::convert_rules(&rule_specs);
    let model = build_model::compute_model(&mut rules, &init_facts);

    eprintln!("DEBUG: computed model with {} atoms", model.len());

    eprintln!("DEBUG: explore_normalized() Step 3: extract fluent facts and functions");
    // Step 4: Extract fluent facts and functions from model
    let fluent_facts = get_fluent_facts(norm_task, &model);
    let fluent_functions = get_fluent_functions(norm_task, &model);
    eprintln!(
        "  Fluent facts: {}, fluent functions: {}",
        fluent_facts.len(),
        fluent_functions.len()
    );

    let fluent_predicates = crate::translate::normalize::get_fluent_predicates(norm_task);
    let init_atom_set = build_init_atom_set(&init_facts);
    let model_atom_set = build_model_atom_set(&model);
    let init_function_values = extract_init_function_values(norm_task);
    let init_predicate_facts = extract_init_predicate_facts(norm_task);
    let type_to_objects = get_objects_by_type(&norm_task.objects, &type_hierarchy);

    eprintln!("DEBUG: explore_normalized() Step 4: ground actions from model");
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

    eprintln!("DEBUG: explore_normalized() Step 5: separate init state into constants and fluents");
    // Step 6: Extract init function values and separate constant facts
    let init_constant_numeric_facts =
        extract_constant_numeric_facts(&init_function_values, &fluent_functions);
    let init_constant_predicate_facts =
        extract_constant_predicate_facts(&init_predicate_facts, &fluent_facts);
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

/// Extract fluent functions (numeric functions) from the model.
/// Matches Python behavior: treat model PNE atoms as fluent functions.
fn get_fluent_functions(
    norm_task: &crate::translate::normalize::NormalizableTask,
    model: &[build_model::Atom],
) -> Vec<String> {
    use std::collections::HashSet;

    let function_symbols: HashSet<String> = norm_task
        .functions
        .iter()
        .map(|(name, _)| name.clone())
        .collect();

    let mut fluent_functions = HashSet::new();
    for atom in model {
        if function_symbols.contains(&atom.predicate) {
            let args = extract_grounded_args(&atom.args);
            fluent_functions.insert(format_pne_key(&atom.predicate, &args));
        }
    }

    fluent_functions.into_iter().collect()
}

fn format_pne_key(name: &str, args: &[String]) -> String {
    if args.is_empty() {
        format!("{}()", name)
    } else {
        format!("{}({})", name, args.join(", "))
    }
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

    init_values
}

fn extract_init_predicate_facts(
    norm_task: &crate::translate::normalize::NormalizableTask,
) -> Vec<build_model::Atom> {
    norm_task
        .init
        .iter()
        .filter_map(|init_sexpr| match init_sexpr {
            SExpr::List(items) if !items.is_empty() => {
                let predicate = match &items[0] {
                    SExpr::Atom(predicate) if predicate != "=" => predicate.clone(),
                    _ => return None,
                };
                let args = items[1..]
                    .iter()
                    .map(|arg| match arg {
                        SExpr::Atom(name) => Some(build_model::Arg::Const(name.clone())),
                        _ => None,
                    })
                    .collect::<Option<Vec<_>>>()?;
                Some(build_model::Atom { predicate, args })
            }
            _ => None,
        })
        .collect()
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
        .filter(|((func_name, args), _value)| {
            !fluent_set.contains(&format_pne_key(func_name, args))
        })
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
) -> Result<
    (
        Vec<GroundedOp>,
        Vec<InstantiatedNumericAxiom>,
        Vec<GroundedAxiom>,
    ),
    InstantiateError,
> {
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
                let uses_metric = norm_task.metric.1.is_some();
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
                    uses_metric,
                )?;
                if let Some(op) = grounded_op {
                    grounded_ops.push(op);
                }
            }
        }
    }

    let fluent_function_set: HashSet<String> = fluent_functions.iter().cloned().collect();

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
                &fluent_function_set,
                init_function_values,
                &mut instantiated_numeric_axioms,
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
            let condition =
                crate::translate::pddl::substitute_condition(&axiom.condition, &variable_mapping);
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
    fluent_function_set: &std::collections::HashSet<String>,
    init_function_values: &std::collections::HashMap<(String, Vec<String>), f64>,
    instantiated_numeric_axioms: &mut std::collections::HashSet<InstantiatedNumericAxiom>,
) -> Option<InstantiatedNumericAxiom> {
    use crate::translate::function_expression::FunctionalExpression;
    use crate::translate::numeric_axiom_rules::{
        InstantiatedNumericAxiom, NumericConstant, NumericPart, PrimitiveNumericExpression,
    };
    use ordered_float::OrderedFloat;

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
                let inst_key_str = format_pne_key(&pne.symbol, &inst_args);
                if !pne.symbol.starts_with("derived!")
                    && !fluent_function_set.contains(&inst_key_str)
                {
                    if let Some(value) = init_function_values.get(&inst_key) {
                        let const_name = format!("derived!{}", format_float(*value));
                        let const_effect = PrimitiveNumericExpression {
                            name: const_name.clone(),
                            args: vec![],
                        };
                        let const_axiom = InstantiatedNumericAxiom {
                            name: format!("({})", const_name),
                            op: None,
                            parts: vec![NumericPart::Constant(NumericConstant(OrderedFloat(
                                *value,
                            )))],
                            effect: const_effect.clone(),
                        };
                        instantiated_numeric_axioms.insert(const_axiom);
                        inst_parts.push(NumericPart::Primitive(const_effect));
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
    uses_metric: bool,
) -> Result<Option<GroundedOp>, InstantiateError> {
    use crate::translate::function_expression::{
        FunctionalExpression, parse_functional_expression,
    };
    use crate::translate::pddl;

    let name = format!("{}({})", action.name, grounded_args.join(","));

    // Substitute variables in precondition
    let pre_sub = pddl::substitute_condition(&action.precondition, variable_mapping);
    let fluent_function_set: std::collections::HashSet<String> =
        fluent_functions.iter().cloned().collect();
    let pre_substituted =
        substitute_condition_numeric(&pre_sub, &fluent_function_set, init_function_values);
    let _preconditions = match instantiate_condition_list(
        &pre_substituted,
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
        Some(pddl::Effect::And(
            effects.iter().map(|(_, e)| e.clone()).collect(),
        ))
    };

    fn extract_cost_expression(
        effect: &crate::translate::pddl_parser::SExpr,
    ) -> Option<crate::translate::pddl_parser::SExpr> {
        if let crate::translate::pddl_parser::SExpr::List(items) = effect {
            if items.len() >= 3 {
                return Some(items[2].clone());
            }
        }
        None
    }

    fn eval_cost_expr(
        expr: &FunctionalExpression,
        var_mapping: &std::collections::HashMap<String, String>,
        init_function_values: &std::collections::HashMap<(String, Vec<String>), f64>,
    ) -> Option<f64> {
        match expr {
            FunctionalExpression::Constant(c) => Some(c.value.into_inner()),
            FunctionalExpression::Primitive(pne) => {
                let args = pne
                    .args
                    .iter()
                    .map(|a| var_mapping.get(a).cloned().unwrap_or_else(|| a.clone()))
                    .collect::<Vec<_>>();
                init_function_values
                    .get(&(pne.symbol.clone(), args))
                    .copied()
            }
            FunctionalExpression::Arithmetic(arith) => {
                let mut values = Vec::new();
                for part in &arith.parts {
                    values.push(eval_cost_expr(part, var_mapping, init_function_values)?);
                }
                match arith.op.as_str() {
                    "+" => Some(values.into_iter().fold(0.0, |acc, v| acc + v)),
                    "-" => {
                        let mut iter = values.into_iter();
                        let first = iter.next()?;
                        Some(iter.fold(first, |acc, v| acc - v))
                    }
                    "*" => Some(values.into_iter().fold(1.0, |acc, v| acc * v)),
                    "/" => {
                        let mut iter = values.into_iter();
                        let first = iter.next()?;
                        let mut acc = first;
                        for v in iter {
                            if v == 0.0 {
                                return None;
                            }
                            acc /= v;
                        }
                        Some(acc)
                    }
                    _ => None,
                }
            }
            FunctionalExpression::AdditiveInverse(inv) => {
                let val = eval_cost_expr(&inv.part, var_mapping, init_function_values)?;
                Some(-val)
            }
        }
    }

    let cost = if uses_metric {
        match &action.cost {
            None => 0.0,
            Some(cost_effect) => {
                if let Some(expr_sexpr) = extract_cost_expression(cost_effect) {
                    if let Some(func_expr) = parse_functional_expression(&expr_sexpr) {
                        eval_cost_expr(&func_expr, variable_mapping, init_function_values)
                            .unwrap_or(1.0)
                    } else {
                        1.0
                    }
                } else {
                    1.0
                }
            }
        }
    } else {
        1.0
    };

    Ok(Some(GroundedOp {
        name,
        args: grounded_args.to_vec(),
        pre: Some(pre_substituted),
        eff: eff_sub,
        effects,
        cost,
    }))
}

fn substitute_condition_numeric(
    condition: &crate::translate::pddl::Condition,
    fluent_function_set: &std::collections::HashSet<String>,
    init_function_values: &std::collections::HashMap<(String, Vec<String>), f64>,
) -> crate::translate::pddl::Condition {
    use crate::translate::pddl::Condition;

    match condition {
        Condition::Comparison(op, left, right) => {
            let left_sub = substitute_sexpr_with_numeric(
                left,
                &std::collections::HashMap::new(),
                fluent_function_set,
                init_function_values,
            )
            .unwrap_or_else(|| left.clone());
            let right_sub = substitute_sexpr_with_numeric(
                right,
                &std::collections::HashMap::new(),
                fluent_function_set,
                init_function_values,
            )
            .unwrap_or_else(|| right.clone());
            Condition::Comparison(op.clone(), left_sub, right_sub)
        }
        Condition::Not(inner) => Condition::Not(Box::new(substitute_condition_numeric(
            inner,
            fluent_function_set,
            init_function_values,
        ))),
        Condition::And(parts) => Condition::And(
            parts
                .iter()
                .map(|p| substitute_condition_numeric(p, fluent_function_set, init_function_values))
                .collect(),
        ),
        Condition::Or(parts) => Condition::Or(
            parts
                .iter()
                .map(|p| substitute_condition_numeric(p, fluent_function_set, init_function_values))
                .collect(),
        ),
        Condition::Forall(params, inner) => Condition::Forall(
            params.clone(),
            Box::new(substitute_condition_numeric(
                inner,
                fluent_function_set,
                init_function_values,
            )),
        ),
        Condition::Exists(params, inner) => Condition::Exists(
            params.clone(),
            Box::new(substitute_condition_numeric(
                inner,
                fluent_function_set,
                init_function_values,
            )),
        ),
        _ => condition.clone(),
    }
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

        let condition = crate::translate::pddl::substitute_condition(&effect.condition, &mapping);
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
                )));
            }
        };
        let parsed = crate::translate::pddl::sexpr_to_effect(&substituted);
        match parsed {
            crate::translate::pddl::Effect::Add(name, args) => {
                if fluent_predicates.contains(&name) {
                    out.push((cond_list, crate::translate::pddl::Effect::Add(name, args)));
                } else if !init_atom_set.contains(&(name.clone(), args.clone())) {
                    return Err(InstantiateError::NonFluentPredicate(name));
                }
            }
            crate::translate::pddl::Effect::Del(name, args) => {
                if fluent_predicates.contains(&name) {
                    out.push((cond_list, crate::translate::pddl::Effect::Del(name, args)));
                } else if init_atom_set.contains(&(name.clone(), args.clone())) {
                    return Err(InstantiateError::NonFluentPredicate(name));
                }
            }
            crate::translate::pddl::Effect::Increase(name, args, val) => {
                if fluent_function_set.contains(&format_pne_key(&name, &args)) {
                    out.push((
                        cond_list,
                        crate::translate::pddl::Effect::Increase(name, args, val),
                    ));
                } else {
                    return Err(InstantiateError::NonFluentFunction(name));
                }
            }
            crate::translate::pddl::Effect::Decrease(name, args, val) => {
                if fluent_function_set.contains(&format_pne_key(&name, &args)) {
                    out.push((
                        cond_list,
                        crate::translate::pddl::Effect::Decrease(name, args, val),
                    ));
                } else {
                    return Err(InstantiateError::NonFluentFunction(name));
                }
            }
            crate::translate::pddl::Effect::And(v) => {
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
                } else if a.starts_with("derived!") {
                    if let Some(val) = init_function_values.get(&(a.clone(), vec![])) {
                        Some(SExpr::Atom(format_number(*val)))
                    } else {
                        Some(SExpr::Atom(a.clone()))
                    }
                } else if fluent_function_set.contains(&format_pne_key(a, &[])) {
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
                if fname.starts_with("derived!") {
                    if let Some(val) = init_function_values.get(&(fname.clone(), args.clone())) {
                        return Some(SExpr::Atom(format_number(*val)));
                    }
                }
                if fluent_function_set.contains(&format_pne_key(fname, &args)) {
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
    condition: &crate::translate::pddl::Condition,
    init_atom_set: &std::collections::HashSet<(String, Vec<String>)>,
    model_atom_set: &std::collections::HashSet<(String, Vec<String>)>,
    fluent_predicates: &std::collections::HashSet<String>,
    fluent_function_set: &std::collections::HashSet<String>,
    init_function_values: &std::collections::HashMap<(String, Vec<String>), f64>,
) -> Option<Vec<Condition>> {
    use crate::translate::pddl::Condition;

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
    format_float(value)
}
