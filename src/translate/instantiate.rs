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

/// High-level exploration step mirroring python/translate/instantiate.py::explore.
///
/// 1. Translate the normalized task into a datalog-style program.
/// 2. Compute its model to discover reachable facts and action instances.
/// 3. Ground operators and numeric axioms using the current Rust substitutes.
pub fn explore(task: &PddlTask) -> ExploreResult {
    // Step 1: translate domain/problem forms to a prolog-style program.
    let prog = pddl_to_prolog::translate_from_ast(&task.domain_forms, &task.problem_forms);

    // Step 2: compute the datalog model (facts reachable under the relaxed semantics).
    let mut rules = build_model::convert_rules(&prog.model_rules);
    let model = build_model::compute_model(&mut rules, &prog.model_facts);

    // Step 3: ground operators and numeric axioms using our current Rust logic.
    let domain =
        Domain::from_sexprs(&task.domain_forms).expect("domain parsing failed during explore");
    let problem =
        Problem::from_sexprs(&task.problem_forms).expect("problem parsing failed during explore");
    let (ops, num_axioms) = ground_with_numeric_axioms(&domain, &problem);

    let relaxed_reachable = model.iter().any(|atom| atom.predicate == "@goal-reachable");

    ExploreResult {
        relaxed_reachable,
        model,
        grounded_ops: ops,
        numeric_axioms: num_axioms,
    }
}
