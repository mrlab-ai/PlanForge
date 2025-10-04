use crate::translate::pddl::{Domain, Problem, Condition, Effect};
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
    for (name, type_str) in &problem.objects {
        by_type.entry(Some(type_str.clone())).or_default().push(name.clone());
        by_type.entry(None).or_default().push(name.clone());
    }

    for action in &domain.actions {
        // for each parameter produce candidate lists
        let mut choices: Vec<Vec<String>> = Vec::new();
        for param in &action.parameters {
            let cands = by_type.get(&param.type_name).cloned().unwrap_or_default();
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
            for (idx, param) in action.parameters.iter().enumerate() {
                mapping.insert(param.name.clone(), args[idx].clone());
            }
            // parse pre/effect into Condition/Effect when possible
            let pre_cond = Some(&action.precondition);
            let eff_e = if action.effects.is_empty() { 
                None 
            } else { 
                Some(&action.effects[0]) 
            };
            // TODO: Implement variable substitution
            // For now, create basic grounded operations without substitution
            result.push(GroundedOp { 
                name, 
                args: args.clone(), 
                pre: pre_cond.cloned(), 
                eff: eff_e.cloned() 
            });
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
    let _df_admin = DerivedFunctionAdministrator::new();
    
    // TODO: Build simple instantiated numeric axioms from numeric init facts
    // This requires converting Literal to SExpr format, which is complex
    let inst_axioms: Vec<crate::translate::numeric_axiom_rules::InstantiatedNumericAxiom> = Vec::new();

    (ops, inst_axioms)
}
