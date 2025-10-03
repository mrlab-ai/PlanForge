use crate::translate::pddl_ast::{Action, Domain, Problem, Condition, Effect};

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
