use crate::translate::build_model;
use crate::translate::derived_function_admin::DerivedFunctionAdministrator;
use crate::translate::numeric_axiom_rules::InstantiatedNumericAxiom;
use crate::translate::pddl::PddlTask;
use crate::translate::pddl_ast::{Condition, Domain, Effect, Problem};
use crate::translate::pddl_to_prolog;

#[derive(Debug, Clone)]
pub struct GroundedOp {
    pub name: String,
    pub args: Vec<String>,
    pub pre: Option<Condition>,
    pub eff: Option<Effect>,
}

/// Naive grounding: for each action, produce substitutions where each parameter
/// is replaced by any object of the matching type (or any object for untyped).
pub fn ground(domain: &Domain, problem: &Problem) -> Vec<GroundedOp> {
    let mut result = Vec::new();
    // prepare objects by type
    use std::collections::HashMap;
    let mut by_type: HashMap<Option<String>, Vec<String>> = HashMap::new();
    for (name, t) in &problem.objects {
        by_type.entry(t.clone()).or_default().push(name.clone());
        by_type.entry(None).or_default().push(name.clone());
    }

    for action in &domain.actions {
        // for each parameter produce candidate lists
        let mut choices: Vec<Vec<String>> = Vec::new();
        for (_pname, ptype) in &action.parameters {
            let cands = by_type.get(ptype).cloned().unwrap_or_default();
            if cands.is_empty() {
                choices.push(vec![]);
            } else {
                choices.push(cands);
            }
        }
        // produce cartesian product
        fn cartesian(v: &[Vec<String>]) -> Vec<Vec<String>> {
            let mut res: Vec<Vec<String>> = vec![Vec::new()];
            for list in v {
                let mut new = Vec::new();
                for prefix in &res {
                    for item in list {
                        let mut np = prefix.clone();
                        np.push(item.clone());
                        new.push(np);
                    }
                }
                res = new;
            }
            res
        }

        for args in cartesian(&choices) {
            let name = format!("{}({})", action.name, args.join(","));
            // build mapping from parameter names to concrete object names
            use std::collections::HashMap;
            let mut mapping: HashMap<String, String> = HashMap::new();
            for (idx, (pname, _)) in action.parameters.iter().enumerate() {
                mapping.insert(pname.clone(), args[idx].clone());
            }
            // parse pre/effect into Condition/Effect when possible
            let pre_cond = action
                .precond
                .as_ref()
                .map(|s| crate::translate::pddl_ast::sexpr_to_condition(s));
            let eff_e = action
                .effect
                .as_ref()
                .map(|s| crate::translate::pddl_ast::sexpr_to_effect(s));
            // substitute variables
            let pre_sub =
                pre_cond.map(|c| crate::translate::pddl_ast::substitute_condition(&c, &mapping));
            let eff_sub =
                eff_e.map(|e| crate::translate::pddl_ast::substitute_effect(&e, &mapping));
            result.push(GroundedOp {
                name,
                args: args.clone(),
                pre: pre_sub,
                eff: eff_sub,
            });
        }
    }

    result
}

/// New API: ground the task and also return instantiated numeric axioms discovered
pub fn ground_with_numeric_axioms(
    domain: &Domain,
    problem: &Problem,
) -> (
    Vec<GroundedOp>,
    Vec<crate::translate::numeric_axiom_rules::InstantiatedNumericAxiom>,
) {
    // For now reuse the same grounding and produce instantiated numeric axioms
    // for numeric init facts. Use the DerivedFunctionAdministrator so the
    // produced PNE names follow the same canonicalization that will be used
    // later during derived-function handling.
    let ops = ground(domain, problem);
    let mut df_admin = DerivedFunctionAdministrator::new();
    // Build simple instantiated numeric axioms from numeric init facts in the problem.
    let mut inst_axioms: Vec<crate::translate::numeric_axiom_rules::InstantiatedNumericAxiom> =
        Vec::new();
    for sexpr in &problem.init {
        // look for forms like (= (f a b) 42)
        if let crate::translate::pddl_parser::SExpr::List(list) = sexpr {
            if list.len() >= 3 {
                if let crate::translate::pddl_parser::SExpr::Atom(eq) = &list[0] {
                    if eq == "=" {
                        if let crate::translate::pddl_parser::SExpr::List(lhs_vec) = &list[1] {
                            // construct an SExpr::List to pass into df_admin
                            let lhs_sexpr =
                                crate::translate::pddl_parser::SExpr::List(lhs_vec.clone());
                            // get canonicalized PNE description (derived! tokens)
                            let pne = df_admin.get_derived_function(&lhs_sexpr);
                            // parse rhs as integer constant if possible
                            if let crate::translate::pddl_parser::SExpr::Atom(rhs) = &list[2] {
                                if let Ok(n) = rhs.parse::<i64>() {
                                    let part = crate::translate::numeric_axiom_rules::NumericPart::Constant(crate::translate::numeric_axiom_rules::NumericConstant(n));
                                    let ax = crate::translate::numeric_axiom_rules::InstantiatedNumericAxiom { name: pne.name.clone(), op: None, parts: vec![part], effect: pne };
                                    inst_axioms.push(ax);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    (ops, inst_axioms)
}

#[derive(Debug, Clone)]
pub struct ExploreResult {
    pub relaxed_reachable: bool,
    pub model: Vec<build_model::Atom>,
    pub grounded_ops: Vec<GroundedOp>,
    pub numeric_axioms: Vec<InstantiatedNumericAxiom>,
}

/// Split a rule with >2 conditions into a chain of binary join/project rules.
/// Mirrors Python's split_rules.py and greedy_join.py logic.
fn split_rule(
    body: Vec<build_model::SymAtom>,
    head: build_model::SymAtom,
    counter: &mut usize,
) -> Vec<(Vec<build_model::SymAtom>, build_model::SymAtom)> {
    if body.len() <= 2 {
        // No splitting needed
        return vec![(body, head)];
    }
    
    // Track which variables are still needed (effect variables + variables in remaining conditions)
    let effect_vars: std::collections::HashSet<String> = 
        head.args.iter().filter(|a| a.starts_with('?')).cloned().collect();
    
    // Greedy join algorithm: repeatedly join the two conditions with minimum cost
    let mut remaining_conditions = body;
    let mut result_rules = Vec::new();
    
    while remaining_conditions.len() > 2 {
        // Find the pair with minimum join cost (most shared variables)
        let (best_i, best_j, _cost) = find_best_join_pair(&remaining_conditions);
        
        // Get variables from the two conditions to join
        let left = &remaining_conditions[best_i];
        let right = &remaining_conditions[best_j];
        let left_vars = get_variables(left);
        let right_vars = get_variables(right);
        let common_vars: std::collections::HashSet<String> = 
            left_vars.iter().filter(|v| right_vars.contains(*v)).cloned().collect();
        
        // Calculate which variables from this join are still needed
        // (needed for effect or for joining with remaining conditions)
        let mut remaining_vars: std::collections::HashSet<String> = effect_vars.clone();
        for (i, cond) in remaining_conditions.iter().enumerate() {
            if i != best_i && i != best_j {
                remaining_vars.extend(get_variables(cond));
            }
        }
        
        // The intermediate atom should only include variables that are still needed
        let joined_vars: std::collections::HashSet<String> = 
            left_vars.iter().chain(right_vars.iter()).cloned().collect();
        let mut intermediate_vars: Vec<String> = joined_vars
            .iter()
            .filter(|v| remaining_vars.contains(*v) || common_vars.contains(*v))
            .cloned()
            .collect();
        intermediate_vars.sort(); // Consistent ordering
        
        // Create intermediate predicate
        let intermediate_name = format!("@new-atom-{}", counter);
        *counter += 1;
        let intermediate_atom = build_model::SymAtom::new(
            intermediate_name,
            intermediate_vars,
        );
        
        // Create join rule for these two conditions
        let join_body = vec![left.clone(), right.clone()];
        result_rules.push((join_body, intermediate_atom.clone()));
        
        // Remove the two joined conditions and add the intermediate
        let mut new_conditions = Vec::new();
        for (i, cond) in remaining_conditions.iter().enumerate() {
            if i != best_i && i != best_j {
                new_conditions.push(cond.clone());
            }
        }
        new_conditions.push(intermediate_atom);
        remaining_conditions = new_conditions;
    }
    
    // Final rule: combine last 2 conditions into the head
    result_rules.push((remaining_conditions, head));
    result_rules
}

/// Find the pair of conditions with the best join cost (most shared variables).
fn find_best_join_pair(conditions: &[build_model::SymAtom]) -> (usize, usize, i32) {
    let mut best_i = 0;
    let mut best_j = 1;
    let mut best_cost = i32::MAX;
    
    for i in 0..conditions.len() {
        for j in (i + 1)..conditions.len() {
            let cost = compute_join_cost(&conditions[i], &conditions[j]);
            if cost < best_cost {
                best_cost = cost;
                best_i = i;
                best_j = j;
            }
        }
    }
    
    (best_i, best_j, best_cost)
}

/// Compute join cost between two conditions.
/// Lower cost = more shared variables (better to join).
fn compute_join_cost(left: &build_model::SymAtom, right: &build_model::SymAtom) -> i32 {
    let left_vars = get_variables(left);
    let right_vars = get_variables(right);
    let common_vars: Vec<_> = left_vars
        .iter()
        .filter(|v| right_vars.contains(*v))
        .collect();
    
    // Cost: prefer joins with more shared variables (negative common count)
    // and fewer unique variables
    let unique_left = left_vars.len() - common_vars.len();
    let unique_right = right_vars.len() - common_vars.len();
    (unique_left + unique_right) as i32 - (common_vars.len() as i32 * 10)
}

/// Get all variables (starting with '?') from an atom.
fn get_variables(atom: &build_model::SymAtom) -> Vec<String> {
    atom.args
        .iter()
        .filter(|a| a.starts_with('?'))
        .cloned()
        .collect()
}

/// High-level exploration step mirroring python/translate/instantiate.py::explore.
///
/// 1. Translate the normalized task into a datalog-style program.
/// 2. Compute its model to discover reachable facts and action instances.
/// 3. Ground operators from model atoms (model-guided, not cartesian product).
/// Explore using a normalized task (preferred).
/// This version builds proper exploration rules from normalized actions.
pub fn explore_normalized(norm_task: &crate::translate::normalize::NormalizableTask) -> ExploreResult {
    eprintln!("DEBUG: explore_normalized() Step 1: build exploration rules");
    // Step 1: Build exploration rules from normalized actions and axioms
    let exploration_rules = crate::translate::normalize::build_exploration_rules(norm_task);
    eprintln!("  Built {} exploration rules", exploration_rules.len());
    
    // Debug: print first 10 rules with full details
    for (i, (body, head)) in exploration_rules.iter().enumerate().take(10) {
        eprintln!("  Rule {}: {}({}) :- [{}]", i, head.predicate, head.args.join(","),
                  body.iter().map(|a| format!("{}({})", a.predicate, a.args.join(","))).collect::<Vec<_>>().join(", "));
    }
    
    eprintln!("DEBUG: explore_normalized() Step 1b: split multi-condition rules");
    // Step 1b: Split rules with >2 conditions into chains of binary joins
    let mut split_rules = Vec::new();
    let mut counter = 0;
    for (body, head) in exploration_rules {
        if body.len() > 2 {
            let sub_rules = split_rule(body, head, &mut counter);
            split_rules.extend(sub_rules);
        } else {
            split_rules.push((body, head));
        }
    }
    eprintln!("  After splitting: {} rules", split_rules.len());
    
    // Debug: print first 25 split rules
    for (i, (body, head)) in split_rules.iter().enumerate().take(25) {
        eprintln!("  Split rule {}: {}({}) :- [{}]", i, head.predicate, head.args.join(","),
                  body.iter().map(|a| format!("{}({})", a.predicate, a.args.join(","))).collect::<Vec<_>>().join(", "));
    }
    
    // Convert to RuleSpec format, separating facts from rules
    let mut rule_specs: Vec<build_model::RuleSpec> = Vec::new();
    let mut extra_facts: Vec<build_model::Atom> = Vec::new();
    
    for (body, head) in split_rules {
        // Determine rule type based on body size and variable sharing
        let rtype = determine_rule_type(&head, &body);
        
        if rtype == "fact" {
            // This is a fact (no body) - add it directly to init facts
            // Convert the head to an atom (no variables, just constants)
            extra_facts.push(build_model::Atom {
                predicate: head.predicate,
                args: head.args.iter().map(|s| build_model::Arg::Const(s.clone())).collect(),
            });
        } else {
            rule_specs.push(build_model::RuleSpec {
                rtype,
                effect: head,
                conditions: body,
            });
        }
    }
    
    eprintln!("DEBUG: explore_normalized() Step 2: add init facts");
    // Step 2: Build init facts from problem
    let mut init_facts: Vec<build_model::Atom> = Vec::new();
    
    // Add type facts for all objects
    for (obj_name, obj_type) in &norm_task.objects {
        if let Some(type_name) = obj_type {
            init_facts.push(build_model::Atom {
                predicate: type_name.clone(),
                args: vec![build_model::Arg::Const(obj_name.clone())],
            });
        }
    }
    
    // Add init state atoms
    for init_sexpr in &norm_task.init {
        if let Some(atom) = sexpr_to_atom(init_sexpr) {
            init_facts.push(atom);
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
    
    eprintln!("DEBUG: explore_normalized() Step 4: ground actions from model");
    // Step 4: Extract grounded actions from model
    let (ops, num_axioms) = ground_from_normalized_model(&model, norm_task);
    eprintln!("DEBUG: grounded {} operators", ops.len());
    
    let relaxed_reachable = model.iter().any(|atom| atom.predicate == "@goal-reachable");
    
    ExploreResult {
        relaxed_reachable,
        model,
        grounded_ops: ops,
        numeric_axioms: num_axioms,
    }
}

/// Ground actions from model using normalized task actions.
fn ground_from_normalized_model(
    model: &[build_model::Atom],
    norm_task: &crate::translate::normalize::NormalizableTask,
) -> (Vec<GroundedOp>, Vec<InstantiatedNumericAxiom>) {
    use std::collections::HashMap;

    // Build action lookup map
    let mut action_map: HashMap<String, &crate::translate::normalize::TaskAction> = HashMap::new();
    for action in &norm_task.actions {
        action_map.insert(action.name.clone(), action);
    }

    let mut grounded_ops = Vec::new();

    // Iterate model atoms and extract action instantiations
    for atom in model {
        // Check if this atom represents an action (predicate starts with @action-)
        if atom.predicate.starts_with("@action-") {
            let action_name = &atom.predicate["@action-".len()..];
            
            if let Some(action) = action_map.get(action_name) {
                // Extract grounded arguments from atom
                let grounded_args = extract_grounded_args(&atom.args);
                
                // Create variable mapping: parameter name -> grounded object
                let variable_mapping = create_variable_mapping(&action.parameters, &grounded_args);
                
                // Instantiate this specific action with these parameters
                let grounded_op = instantiate_normalized_action(action, &grounded_args, &variable_mapping);
                
                grounded_ops.push(grounded_op);
            }
        }
    }

    // For now, return empty numeric axioms (will be filled in later phases)
    let num_axioms = Vec::new();

    (grounded_ops, num_axioms)
}

/// Instantiate a normalized action with specific grounded arguments.
fn instantiate_normalized_action(
    action: &crate::translate::normalize::TaskAction,
    grounded_args: &[String],
    variable_mapping: &std::collections::HashMap<String, String>,
) -> GroundedOp {
    use crate::translate::pddl_ast;
    
    let name = format!("{}({})", action.name, grounded_args.join(","));
    
    // Substitute variables in precondition
    let pre_sub = pddl_ast::substitute_condition(&action.precondition, variable_mapping);
    
    // For effects, we need to convert from TaskEffect to Effect
    // For now, use a simplified approach (will be enhanced in Phase 3)
    let eff_sub = None; // Placeholder - full effect handling in Phase 3
    
    GroundedOp {
        name,
        args: grounded_args.to_vec(),
        pre: Some(pre_sub),
        eff: eff_sub,
    }
}

/// Determine the rule type (join/product/project) based on conditions and effect.
fn determine_rule_type(_head: &build_model::SymAtom, body: &[build_model::SymAtom]) -> String {
    match body.len() {
        0 => {
            // No conditions - this becomes a fact
            "fact".to_string()
        }
        1 => "project".to_string(), // Single condition - projection
        2 => {
            // For 2 conditions, check if there are shared variables
            let vars_1: std::collections::HashSet<&String> = body[0]
                .args
                .iter()
                .filter(|a| a.starts_with('?'))
                .collect();
            let vars_2: std::collections::HashSet<&String> = body[1]
                .args
                .iter()
                .filter(|a| a.starts_with('?'))
                .collect();
            
            if vars_1.intersection(&vars_2).count() > 0 {
                "join".to_string()
            } else {
                "product".to_string()
            }
        }
        _ => {
            // For >2 conditions, use "product"
            // Note: JoinRule only supports 2 conditions, so we can't use "join" here
            // even if variables are shared. ProductRule will handle variable constraints.
            "product".to_string()
        }
    }
}

/// Convert a PDDL SExpr to a build_model Atom.
fn sexpr_to_atom(sexpr: &crate::translate::pddl_parser::SExpr) -> Option<build_model::Atom> {
    use crate::translate::pddl_parser::SExpr;
    match sexpr {
        SExpr::List(items) if !items.is_empty() => {
            if let SExpr::Atom(pred) = &items[0] {
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

pub fn explore(task: &PddlTask) -> ExploreResult {
    eprintln!("DEBUG: explore() Step 1: translate to prolog");
    // Step 1: translate domain/problem forms to a prolog-style program.
    let prog = pddl_to_prolog::translate_from_ast(&task.domain_forms, &task.problem_forms);

    eprintln!("DEBUG: explore() Step 2: compute model");
    // Step 2: compute the datalog model (facts reachable under the relaxed semantics).
    let mut rules = build_model::convert_rules(&prog.model_rules);
    let model = build_model::compute_model(&mut rules, &prog.model_facts);
    eprintln!("DEBUG: computed model with {} atoms", model.len());
    eprintln!("DEBUG: model facts (first 50):");
    for (i, atom) in model.iter().take(50).enumerate() {
        eprintln!("  {}: {}({:?})", i + 1, atom.predicate, atom.args);
    }
    eprintln!("DEBUG: checking for @action- prefixed atoms:");
    let action_atoms: Vec<_> = model.iter()
        .filter(|a| a.predicate.starts_with("@action-"))
        .collect();
    eprintln!("  Found {} action atoms", action_atoms.len());
    for atom in &action_atoms {
        eprintln!("    - {}({:?})", atom.predicate, atom.args);
    }

    eprintln!("DEBUG: explore() Step 3: parse domain and problem");
    // Step 3: Parse domain and problem for action definitions
    let domain =
        Domain::from_sexprs(&task.domain_forms).expect("domain parsing failed during explore");
    let problem =
        Problem::from_sexprs(&task.problem_forms).expect("problem parsing failed during explore");
    eprintln!("DEBUG: parsed {} actions", domain.actions.len());

    eprintln!("DEBUG: explore() Step 4: model-guided grounding");
    // Step 4: Model-guided grounding - instantiate actions from model atoms
    let (ops, num_axioms) = ground_from_model(&model, &domain, &problem);
    eprintln!("DEBUG: grounded {} operators", ops.len());

    let relaxed_reachable = model.iter().any(|atom| atom.predicate == "@goal-reachable");

    ExploreResult {
        relaxed_reachable,
        model,
        grounded_ops: ops,
        numeric_axioms: num_axioms,
    }
}

/// Ground actions from model atoms (model-guided instantiation).
/// This replaces the naive cartesian product approach.
fn ground_from_model(
    model: &[build_model::Atom],
    domain: &Domain,
    problem: &Problem,
) -> (Vec<GroundedOp>, Vec<InstantiatedNumericAxiom>) {
    use std::collections::HashMap;

    // Build action lookup map
    let mut action_map: HashMap<String, &crate::translate::pddl_ast::Action> = HashMap::new();
    for action in &domain.actions {
        action_map.insert(action.name.clone(), action);
    }

    let mut grounded_ops = Vec::new();

    // Iterate model atoms and extract action instantiations
    for atom in model {
        // Check if this atom represents an action (predicate starts with @action-)
        if atom.predicate.starts_with("@action-") {
            let action_name = &atom.predicate["@action-".len()..];
            
            if let Some(action) = action_map.get(action_name) {
                // Extract grounded arguments from atom
                let grounded_args = extract_grounded_args(&atom.args);
                
                // Create variable mapping: parameter name -> grounded object
                let variable_mapping = create_variable_mapping(&action.parameters, &grounded_args);
                
                // Instantiate this specific action with these parameters
                let grounded_op = instantiate_action_simple(action, &grounded_args, &variable_mapping);
                
                grounded_ops.push(grounded_op);
            }
        }
    }

    // Build numeric axioms from problem init (same as before)
    let mut df_admin = DerivedFunctionAdministrator::new();
    let mut inst_axioms: Vec<InstantiatedNumericAxiom> = Vec::new();
    
    for sexpr in &problem.init {
        if let crate::translate::pddl_parser::SExpr::List(list) = sexpr {
            if list.len() >= 3 {
                if let crate::translate::pddl_parser::SExpr::Atom(eq) = &list[0] {
                    if eq == "=" {
                        if let crate::translate::pddl_parser::SExpr::List(lhs_vec) = &list[1] {
                            let lhs_sexpr = crate::translate::pddl_parser::SExpr::List(lhs_vec.clone());
                            let pne = df_admin.get_derived_function(&lhs_sexpr);
                            
                            if let crate::translate::pddl_parser::SExpr::Atom(rhs) = &list[2] {
                                if let Ok(n) = rhs.parse::<i64>() {
                                    let part = crate::translate::numeric_axiom_rules::NumericPart::Constant(
                                        crate::translate::numeric_axiom_rules::NumericConstant(n)
                                    );
                                    let ax = InstantiatedNumericAxiom {
                                        name: pne.name.clone(),
                                        op: None,
                                        parts: vec![part],
                                        effect: pne,
                                    };
                                    inst_axioms.push(ax);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    (grounded_ops, inst_axioms)
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

/// Instantiate an action with specific grounded arguments (simple version for Phase 1).
/// This performs basic variable substitution. Phase 3 will add proper condition evaluation.
fn instantiate_action_simple(
    action: &crate::translate::pddl_ast::Action,
    grounded_args: &[String],
    variable_mapping: &std::collections::HashMap<String, String>,
) -> GroundedOp {
    let name = format!("{}({})", action.name, grounded_args.join(","));
    
    // Substitute variables in precondition
    let pre_cond = action
        .precond
        .as_ref()
        .map(|s| crate::translate::pddl_ast::sexpr_to_condition(s));
    let pre_sub = pre_cond.map(|c| crate::translate::pddl_ast::substitute_condition(&c, variable_mapping));
    
    // Substitute variables in effects
    let eff_e = action
        .effect
        .as_ref()
        .map(|s| crate::translate::pddl_ast::sexpr_to_effect(s));
    let eff_sub = eff_e.map(|e| crate::translate::pddl_ast::substitute_effect(&e, variable_mapping));
    
    GroundedOp {
        name,
        args: grounded_args.to_vec(),
        pre: pre_sub,
        eff: eff_sub,
    }
}
