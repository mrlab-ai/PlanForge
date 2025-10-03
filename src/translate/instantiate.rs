use crate::translate::pddl_ast::{Action, Domain, Problem, Condition, Effect};
use crate::translate::derived_function_admin::DerivedFunctionAdministrator;

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
            let pre_cond = action.precond.as_ref().map(|s| {
                crate::translate::pddl_ast::sexpr_to_condition(s)
            });
            let eff_e = action.effect.as_ref().map(|s| {
                crate::translate::pddl_ast::sexpr_to_effect(s)
            });
            // substitute variables
            let pre_sub = pre_cond.map(|c| crate::translate::pddl_ast::substitute_condition(&c, &mapping));
            let eff_sub = eff_e.map(|e| crate::translate::pddl_ast::substitute_effect(&e, &mapping));
            result.push(GroundedOp { name, args: args.clone(), pre: pre_sub, eff: eff_sub });
        }
    }

    result
}

/// New API: ground the task and also return instantiated numeric axioms discovered
pub fn ground_with_numeric_axioms(domain: &Domain, problem: &Problem) -> (Vec<GroundedOp>, Vec<crate::translate::numeric_axiom_rules::InstantiatedNumericAxiom>) {
    // For now reuse the same grounding and produce instantiated numeric axioms
    // for numeric init facts. Use the DerivedFunctionAdministrator so the
    // produced PNE names follow the same canonicalization that will be used
    // later during derived-function handling.
    let ops = ground(domain, problem);
    let mut df_admin = DerivedFunctionAdministrator::new();
    // Build simple instantiated numeric axioms from numeric init facts in the problem.
    let mut inst_axioms: Vec<crate::translate::numeric_axiom_rules::InstantiatedNumericAxiom> = Vec::new();
    for sexpr in &problem.init {
        // look for forms like (= (f a b) 42)
        if let crate::translate::pddl_parser::SExpr::List(list) = sexpr {
            if list.len() >= 3 {
                if let crate::translate::pddl_parser::SExpr::Atom(eq) = &list[0] {
                    if eq == "=" {
                        if let crate::translate::pddl_parser::SExpr::List(lhs_vec) = &list[1] {
                            // construct an SExpr::List to pass into df_admin
                            let lhs_sexpr = crate::translate::pddl_parser::SExpr::List(lhs_vec.clone());
                            // get canonicalized PNE description
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
