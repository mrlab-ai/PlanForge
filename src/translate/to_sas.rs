use crate::translate::instantiate::GroundedOp;
use crate::translate::pddl_ast::{Domain, Problem};
use crate::translate::sas::SASTask;

// Helper to ensure a numeric variable exists for an expression. This is a
// module-level function so we can pass a mutable DerivedFunctionAdministrator
// reference without closure capture problems.
fn ensure_expr_var_visit(
    sexpr: &crate::translate::pddl_parser::SExpr,
    df_admin: &mut crate::translate::derived_function_admin::DerivedFunctionAdministrator,
    num_index: &mut std::collections::HashMap<String, usize>,
    numeric_list: &mut Vec<crate::translate::sas::NumericVariable>,
    numeric_init_vec: &mut Vec<i64>,
    instantiated_num_axioms: &mut Vec<
        crate::translate::numeric_axiom_rules::InstantiatedNumericAxiom,
    >,
    derived_axiom_index: &mut std::collections::HashMap<String, usize>,
) -> Option<String> {
    match sexpr {
        crate::translate::pddl_parser::SExpr::List(inner) => {
            if inner.is_empty() {
                return None;
            }
            if let crate::translate::pddl_parser::SExpr::Atom(op) = &inner[0] {
                if op == "+" || op == "-" || op == "*" || op == "/" {
                    // Ask df_admin for canonical operator name and build parts
                    let pne = df_admin.get_derived_function(
                        &crate::translate::pddl_parser::SExpr::List(inner.clone()),
                    );
                    let derived_name = if pne.args.is_empty() {
                        pne.name.clone()
                    } else {
                        format!("{} {}", pne.name, pne.args.join(" "))
                    };
                    let mut parts_numericparts: Vec<
                        crate::translate::numeric_axiom_rules::NumericPart,
                    > = Vec::new();
                    for p in &inner[1..] {
                        match p {
                            crate::translate::pddl_parser::SExpr::Atom(a) => {
                                if let Ok(nv) = a.parse::<i64>() {
                                    parts_numericparts.push(crate::translate::numeric_axiom_rules::NumericPart::Constant(crate::translate::numeric_axiom_rules::NumericConstant(nv)));
                                } else {
                                    let prim = crate::translate::numeric_axiom_rules::PrimitiveNumericExpression { name: a.clone(), args: vec![] };
                                    parts_numericparts.push(crate::translate::numeric_axiom_rules::NumericPart::Primitive(prim));
                                }
                            }
                            crate::translate::pddl_parser::SExpr::List(_) => {
                                let child = df_admin.get_derived_function(p);
                                let prim = crate::translate::numeric_axiom_rules::PrimitiveNumericExpression { name: child.name.clone(), args: child.args.clone() };
                                parts_numericparts.push(
                                    crate::translate::numeric_axiom_rules::NumericPart::Primitive(
                                        prim,
                                    ),
                                );
                            }
                        }
                    }
                    if !num_index.contains_key(&derived_name) {
                        let idx = numeric_list.len();
                        num_index.insert(derived_name.clone(), idx);
                        numeric_list.push(crate::translate::sas::NumericVariable {
                            name: derived_name.clone(),
                            initial: None,
                            ntype: "D".to_string(),
                            axiom_layer: -1,
                        });
                        numeric_init_vec.push(0);
                        let effect =
                            crate::translate::numeric_axiom_rules::PrimitiveNumericExpression {
                                name: pne.name.clone(),
                                args: pne.args.clone(),
                            };
                        let ax = crate::translate::numeric_axiom_rules::InstantiatedNumericAxiom {
                            name: derived_name.clone(),
                            op: Some(op.clone()),
                            parts: parts_numericparts,
                            effect,
                        };
                        let ai = instantiated_num_axioms.len();
                        instantiated_num_axioms.push(ax.clone());
                        derived_axiom_index.insert(derived_name.clone(), ai);
                    }
                    return Some(derived_name);
                }
                // not an arithmetic op -> treat as PNE (list with fname and args)
                if let crate::translate::pddl_parser::SExpr::Atom(fname) = &inner[0] {
                    let args = inner[1..]
                        .iter()
                        .filter_map(|x| match x {
                            crate::translate::pddl_parser::SExpr::Atom(a) => Some(a.clone()),
                            _ => None,
                        })
                        .collect::<Vec<_>>();
                    let key = format!("{}({})", fname, args.join(", "));
                    return Some(key);
                }
                None
            } else {
                None
            }
        }
        crate::translate::pddl_parser::SExpr::Atom(a) => {
            if let Ok(v) = a.parse::<i64>() {
                // Create a derived constant PNE like Python does: derived!{value}.0()
                // This ensures constants are registered as proper numeric variables
                let const_pne_name = format!("derived!{}.0()", v);
                if !num_index.contains_key(&const_pne_name) {
                    let idx = numeric_list.len();
                    num_index.insert(const_pne_name.clone(), idx);
                    numeric_list.push(crate::translate::sas::NumericVariable {
                        name: const_pne_name.clone(),
                        initial: Some(v),
                        ntype: "C".to_string(),
                        axiom_layer: -1,
                    });
                    numeric_init_vec.push(v);
                    // Create an InstantiatedNumericAxiom for the constant
                    let pne = crate::translate::numeric_axiom_rules::PrimitiveNumericExpression {
                        name: const_pne_name.clone(),
                        args: vec![],
                    };
                    let part = crate::translate::numeric_axiom_rules::NumericPart::Constant(
                        crate::translate::numeric_axiom_rules::NumericConstant(v),
                    );
                    let ax = crate::translate::numeric_axiom_rules::InstantiatedNumericAxiom {
                        name: const_pne_name.clone(),
                        op: None,
                        parts: vec![part],
                        effect: pne,
                    };
                    let ai = instantiated_num_axioms.len();
                    instantiated_num_axioms.push(ax);
                    derived_axiom_index.insert(const_pne_name.clone(), ai);
                }
                Some(const_pne_name)
            } else {
                None
            }
        }
    }
}

/// Build boolean variables for each grounded atom occurring in init/pre/effects.
pub fn build_sas(
    ops: &[GroundedOp],
    dom: &Domain,
    prob: &Problem,
    external_instantiated_num_axioms: &Vec<
        crate::translate::numeric_axiom_rules::InstantiatedNumericAxiom,
    >,
    py_groups: Option<Vec<Vec<String>>>,
) -> SASTask {
    fn normalize_op(op: &str) -> String {
        match op {
            "=" => "eq".to_string(),
            "<=" => "le".to_string(),
            ">=" => "ge".to_string(),
            "<" => "lt".to_string(),
            ">" => "gt".to_string(),
            other => other.to_string(),
        }
    }
    use std::collections::HashMap;
    let mut var_index: HashMap<String, usize> = HashMap::new();
    // mapping from grounded positive atom -> (fdr_var_index, value_index)
    let mut atom_to_fdr: std::collections::HashMap<String, (usize, usize)> =
        std::collections::HashMap::new();
    let mut vars: Vec<crate::translate::sas::Variable> = Vec::new();
    let mut ranges: Vec<usize> = Vec::new();
    let mut fact_to_varvals: HashMap<String, Vec<(usize, usize)>> = HashMap::new();
    // helper removed: add_comparison_var was unused
    // collect instantiated numeric axioms discovered while processing expressions
    let mut instantiated_num_axioms: Vec<
        crate::translate::numeric_axiom_rules::InstantiatedNumericAxiom,
    > = external_instantiated_num_axioms.clone();
    // helper: given an SExpr that may be an arithmetic expression like
    // (+ (f a) (g b)) produce or reuse a numeric variable name, possibly
    // creating an InstantiatedNumericAxiom for the derived expression.
    // Map of derived expression name -> index into instantiated_num_axioms
    let mut derived_axiom_index: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();

    // recursive helper: ensure a numeric variable exists for the expression; returns its canonical name
    let mut df_admin =
        crate::translate::derived_function_admin::DerivedFunctionAdministrator::new();
    let mut ensure_expr_var =
        |sexpr: &crate::translate::pddl_parser::SExpr,
         num_index: &mut std::collections::HashMap<String, usize>,
         numeric_list: &mut Vec<crate::translate::sas::NumericVariable>,
         numeric_init_vec: &mut Vec<i64>,
         instantiated_num_axioms: &mut Vec<
            crate::translate::numeric_axiom_rules::InstantiatedNumericAxiom,
        >,
         derived_axiom_index: &mut std::collections::HashMap<String, usize>|
         -> Option<String> {
            ensure_expr_var_visit(
                sexpr,
                &mut df_admin,
                num_index,
                numeric_list,
                numeric_init_vec,
                instantiated_num_axioms,
                derived_axiom_index,
            )
        };
    // Determine fluent predicates (those that appear in add/del effects) so we can
    // exclude static predicates like door/mount from variables and prevails.
    let mut fluent_preds: std::collections::HashSet<String> = std::collections::HashSet::new();
    for act in &dom.actions {
        if let Some(eff_s) = &act.effect {
            let eff = crate::translate::pddl_ast::sexpr_to_effect(eff_s);
            fn collect(
                e: &crate::translate::pddl_ast::Effect,
                set: &mut std::collections::HashSet<String>,
            ) {
                match e {
                    crate::translate::pddl_ast::Effect::Add(n, _)
                    | crate::translate::pddl_ast::Effect::Del(n, _) => {
                        set.insert(n.clone());
                    }
                    crate::translate::pddl_ast::Effect::And(v) => {
                        for sub in v {
                            collect(sub, set);
                        }
                    }
                    _ => {}
                }
            }
            collect(&eff, &mut fluent_preds);
        }
    }

    // collect numeric inits and boolean init atoms (fluent only)
    let mut numeric_inits: Vec<(String, i64)> = Vec::new();
    // temporary set of grounded boolean atoms encountered
    let mut grounded_atoms: Vec<String> = Vec::new();
    // set of function names declared in the domain so we can distinguish numeric inits
    let func_names: std::collections::HashSet<String> =
        dom.functions.iter().map(|(n, _)| n.clone()).collect();
    for a in &prob.init {
        if let crate::translate::pddl_parser::SExpr::List(list) = a {
            if list.len() >= 3 {
                if let crate::translate::pddl_parser::SExpr::Atom(op) = &list[0] {
                    if op == "=" {
                        if let crate::translate::pddl_parser::SExpr::List(left) = &list[1] {
                            if let crate::translate::pddl_parser::SExpr::Atom(fname) = &left[0] {
                                let arg_s = left[1..]
                                    .iter()
                                    .filter_map(|x| match x {
                                        crate::translate::pddl_parser::SExpr::Atom(s) => {
                                            Some(s.clone())
                                        }
                                        _ => None,
                                    })
                                    .collect::<Vec<_>>()
                                    .join(", ");
                                let key = format!("{}({})", fname, arg_s);
                                if let crate::translate::pddl_parser::SExpr::Atom(val) = &list[2] {
                                    if let Ok(n) = val.parse::<i64>() {
                                        // If lhs is a numeric function declared in the domain, treat as numeric init
                                        if func_names.contains(fname) {
                                            numeric_inits.push((key, n));
                                        } else {
                                            // otherwise keep as a grounded boolean atom as before
                                            grounded_atoms.push(key.clone());
                                            numeric_inits.push((key, n));
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        // boolean init atom
                        if let crate::translate::pddl_parser::SExpr::Atom(name) = &list[0] {
                            if fluent_preds.contains(name) {
                                let atom = format!(
                                    "{}({})",
                                    name,
                                    list[1..]
                                        .iter()
                                        .filter_map(|x| match x {
                                            crate::translate::pddl_parser::SExpr::Atom(s) =>
                                                Some(s.clone()),
                                            _ => None,
                                        })
                                        .collect::<Vec<_>>()
                                        .join(", ")
                                );
                                grounded_atoms.push(atom.clone());
                            }
                        }
                    }
                }
            }
        }
    }

    // First pass: collect atoms from ops and numeric var effect hints
    let mut numeric_vars: Vec<(String, i64)> = Vec::new();
    // helper to canonicalize numeric variable names to the PNE format used elsewhere
    let canon_num = |name: &str| -> String {
        if name.contains('(') {
            name.to_string()
        } else {
            format!("{}()", name)
        }
    };
    for op in ops {
        if let Some(eff) = &op.eff {
            match eff {
                crate::translate::pddl_ast::Effect::Add(name, args) => {
                    grounded_atoms.push(format!("{}({})", name, args.join(", ")));
                }
                crate::translate::pddl_ast::Effect::Del(name, args) => {
                    grounded_atoms.push(format!("{}({})", name, args.join(", ")));
                }
                crate::translate::pddl_ast::Effect::And(v) => {
                    for sub in v {
                        match sub {
                            crate::translate::pddl_ast::Effect::Add(name, args) => {
                                grounded_atoms.push(format!("{}({})", name, args.join(", ")));
                            }
                            crate::translate::pddl_ast::Effect::Del(name, args) => {
                                grounded_atoms.push(format!("{}({})", name, args.join(", ")));
                            }
                            crate::translate::pddl_ast::Effect::Increase(name, _args, val) => {
                                numeric_vars.push((canon_num(name), *val));
                            }
                            crate::translate::pddl_ast::Effect::Decrease(name, _args, val) => {
                                numeric_vars.push((canon_num(name), -*val));
                            }
                            crate::translate::pddl_ast::Effect::And(_) => { /* nested And - ignore for now */
                            }
                        }
                    }
                }
                crate::translate::pddl_ast::Effect::Increase(name, _args, v) => {
                    numeric_vars.push((canon_num(name), *v));
                }
                crate::translate::pddl_ast::Effect::Decrease(name, _args, v) => {
                    numeric_vars.push((canon_num(name), -*v));
                }
            }
        }
        if let Some(pre) = &op.pre {
            match pre {
                crate::translate::pddl_ast::Condition::Atom(name, args) => {
                    if fluent_preds.contains(name) {
                        grounded_atoms.push(format!("{}({})", name, args.join(", ")));
                    }
                }
                crate::translate::pddl_ast::Condition::And(v) => {
                    for c in v {
                        match c {
                            crate::translate::pddl_ast::Condition::Atom(name, args) => {
                                if fluent_preds.contains(name) {
                                    grounded_atoms.push(format!("{}({})", name, args.join(", ")));
                                }
                            }
                            crate::translate::pddl_ast::Condition::Comparison(_, _, _) => {}
                            crate::translate::pddl_ast::Condition::Not(_) => {}
                            crate::translate::pddl_ast::Condition::And(_) => {}
                            crate::translate::pddl_ast::Condition::Or(_) => {}
                            crate::translate::pddl_ast::Condition::Forall(_, _) => {}
                            crate::translate::pddl_ast::Condition::Exists(_, _) => {}
                            crate::translate::pddl_ast::Condition::True => {}
                        }
                    }
                }
                crate::translate::pddl_ast::Condition::Comparison(_, _, _) => { /* handled later */
                }
                crate::translate::pddl_ast::Condition::Not(_) => { /* ignore */ }
                crate::translate::pddl_ast::Condition::Or(_) => { /* should be normalized */ }
                crate::translate::pddl_ast::Condition::Forall(_, _) => { /* should be normalized */ }
                crate::translate::pddl_ast::Condition::Exists(_, _) => { /* should be normalized */ }
                crate::translate::pddl_ast::Condition::True => { /* ignore */ }
            }
        }
    }

    // Compute fact groups: prefer externally provided Python groups for faithful semantics,
    // otherwise use the Rust port of invariant-based grouping.
    let (chosen_groups, _mutex_groups, translation_key) = if let Some(pg) = py_groups {
        (Vec::new(), Vec::new(), pg)
    } else {
        crate::translate::fact_groups::compute_groups(dom, prob, &grounded_atoms, None)
    };
    // Build lookup tables for propositional facts based on translation_key
    for (var_no, group_values) in translation_key.iter().enumerate() {
        ranges.push(group_values.len());
        for (val_no, fact) in group_values.iter().enumerate() {
            fact_to_varvals
                .entry(fact.clone())
                .or_default()
                .push((var_no, val_no));
        }
    }
    for (fact, entries) in &fact_to_varvals {
        if let Some(&(var_idx, val_idx)) = entries.first() {
            if !fact.starts_with("<")
                && !fact.starts_with("NegatedAtom ")
                && !fact.starts_with("not ")
            {
                atom_to_fdr.insert(fact.clone(), (var_idx, val_idx));
            }
        }
    }

    // Build variables from translation_key
    for (var_no, group_values) in translation_key.iter().enumerate() {
        let mut value_names: Vec<String> = Vec::new();
        for v in group_values {
            if v.starts_with("<") || v.starts_with("NegatedAtom ") || v.starts_with("not ") {
                value_names.push(v.clone());
            } else {
                value_names.push(format!("Atom {}", v));
            }
        }
        debug_assert_eq!(value_names.len(), ranges[var_no]);
        vars.push(crate::translate::sas::Variable {
            value_names: value_names.clone(),
        });
    }
    // Build mutex groups from chosen_groups: each group is a list of facts (var,val)
    let mut mutex_groups_pairs: Vec<Vec<(usize, usize)>> = Vec::new();
    if !chosen_groups.is_empty() {
        for group in chosen_groups {
            // translate each atom string to (var,val)
            let mg: Vec<(usize, usize)> = group
                .into_iter()
                .filter_map(|atom| atom_to_fdr.get(&atom).cloned())
                .collect();
            // keep only groups with at least two facts
            if mg.len() > 1 {
                mutex_groups_pairs.push(mg);
            }
        }
    }
    // numeric effect hints already collected above

    // fold numeric vars into NumericVariable structs; prefer init values from problem init, otherwise use effect values
    let mut num_map: std::collections::HashMap<String, i64> = std::collections::HashMap::new();
    for (n, v) in numeric_vars.into_iter() {
        num_map.entry(n).or_insert(v);
    }
    // override with numeric_inits from problem init if present
    for (k, v) in numeric_inits.into_iter() {
        num_map.insert(k, v);
    }
    // Ensure cost() exists (initialize to 0 if absent) and total-cost() exists for metric
    num_map.entry("cost()".to_string()).or_insert(0);
    num_map.entry("total-cost()".to_string()).or_insert(0);

    let mut numeric_list: Vec<crate::translate::sas::NumericVariable> = Vec::new();
    let mut numeric_init_vec: Vec<i64> = Vec::new();
    // deterministic ordering of numeric variables: sort by name
    let mut num_entries: Vec<(String, i64)> =
        num_map.iter().map(|(k, v)| (k.clone(), *v)).collect();
    num_entries.sort_by(|a, b| a.0.cmp(&b.0));
    for (n, v) in num_entries.iter() {
        // heuristically classify numeric variable type: constants 'C', derived 'D' if expression or contains '+'/'-' or 'derived' patterns, regular 'R', instrumentation 'I'
        let ntype = if n == "total-cost()" {
            "I".to_string()
        } else if n.contains("derived!") || (n.contains('+') || n.contains('-')) && n.contains('(')
        {
            "D".to_string()
        } else if n.starts_with("const:") {
            "C".to_string()
        } else if n == "cost()" || n.ends_with("cost()") {
            "R".to_string()
        } else {
            "R".to_string()
        };
        numeric_list.push(crate::translate::sas::NumericVariable {
            name: n.clone(),
            initial: Some(*v),
            ntype,
            axiom_layer: -1,
        });
        numeric_init_vec.push(*v);
    }

    let mut metric_idx: isize = -1;
    for (i, nv) in numeric_list.iter().enumerate() {
        if nv.name == "total-cost()" {
            metric_idx = i as isize;
            break;
        }
        if metric_idx == -1 && nv.name == "cost()" {
            metric_idx = i as isize;
        }
    }

    // build mapping from numeric name to index
    let mut num_index: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for (i, nv) in numeric_list.iter().enumerate() {
        num_index.insert(nv.name.clone(), i);
    }

    // second pass: build operators with prevails/effects and numeric_effects
    let mut operators: Vec<crate::translate::sas::SASOperator> = Vec::new();
    let mut numeric_axioms: Vec<crate::translate::sas::NumericAxiom> = Vec::new();
    let mut comp_axioms: Vec<crate::translate::sas::CompareAxiom> = Vec::new();
    let _comp_index: std::collections::HashMap<(String, Vec<usize>, usize), usize> =
        std::collections::HashMap::new();
    for op in ops {
        let mut prevails: Vec<(usize, usize)> = Vec::new();
        // effects: (var, pre, post, condition) where pre is -1 if no precondition
        let mut effects: Vec<(usize, usize, usize, Vec<(usize, usize)>)> = Vec::new();
        // numeric_effects: (nvar, op, rhs_var, condition)
        let mut op_numeric_effects: Vec<(usize, String, usize, Vec<(usize, usize)>)> = Vec::new();

        if let Some(pre) = &op.pre {
            match pre {
                crate::translate::pddl_ast::Condition::Atom(name, args) => {
                    let atom = format!("{}({})", name, args.join(", "));
                    if let Some(&(v, val)) = atom_to_fdr.get(&atom) {
                        prevails.push((v, val));
                    }
                }
                crate::translate::pddl_ast::Condition::And(v) => {
                    for c in v {
                        match c {
                            crate::translate::pddl_ast::Condition::Atom(name, args) => {
                                let atom = format!("{}({})", name, args.join(", "));
                                if let Some(&(v, val)) = atom_to_fdr.get(&atom) {
                                    prevails.push((v, val));
                                }
                            }
                            crate::translate::pddl_ast::Condition::Comparison(opstr, l, r) => {
                                // Use ensure_expr_var on both operands to get their numeric variable names
                                // This handles PNEs, arithmetic expressions, and constants (which become derived!{v}.0())
                                let left_name = ensure_expr_var(
                                    l,
                                    &mut num_index,
                                    &mut numeric_list,
                                    &mut numeric_init_vec,
                                    &mut instantiated_num_axioms,
                                    &mut derived_axiom_index,
                                );
                                let right_name = ensure_expr_var(
                                    r,
                                    &mut num_index,
                                    &mut numeric_list,
                                    &mut numeric_init_vec,
                                    &mut instantiated_num_axioms,
                                    &mut derived_axiom_index,
                                );
                                if let (Some(left_key), Some(right_key)) = (left_name, right_name) {
                                    if let (Some(&ni), Some(&nj)) = (num_index.get(&left_key), num_index.get(&right_key)) {
                                        let comp_key = format!("{} {} {}", opstr, left_key, right_key);
                                        let effect_idx = if let Some(&ei) = var_index.get(&comp_key) {
                                            ei
                                        } else {
                                            let ei = vars.len();
                                            var_index.insert(comp_key.clone(), ei);
                                            let pos = format!("{} {} {}", opstr, left_key, right_key);
                                            let neg = format!("not {} {} {}", opstr, left_key, right_key);
                                            vars.push(crate::translate::sas::Variable {
                                                value_names: vec![pos, neg, "<none of those>".to_string()],
                                            });
                                            ranges.push(3);
                                            ei
                                        };
                                        comp_axioms.push(crate::translate::sas::CompareAxiom {
                                            comp: normalize_op(opstr),
                                            parts: vec![ni, nj],
                                            effect_var: effect_idx,
                                        });
                                    }
                                }
                            }
                            crate::translate::pddl_ast::Condition::Not(_) => { /* ignore */ }
                            crate::translate::pddl_ast::Condition::And(_) => { /* ignore */ }
                            crate::translate::pddl_ast::Condition::Or(_) => { /* should be normalized */ }
                            crate::translate::pddl_ast::Condition::Forall(_, _) => { /* should be normalized */ }
                            crate::translate::pddl_ast::Condition::Exists(_, _) => { /* should be normalized */ }
                            crate::translate::pddl_ast::Condition::True => { /* ignore */ }
                        }
                    }
                }
                crate::translate::pddl_ast::Condition::Comparison(opstr, l, r) => {
                    // Use ensure_expr_var on both operands to get their numeric variable names
                    // This handles PNEs, arithmetic expressions, and constants (which become derived!{v}.0())
                    let left_name = ensure_expr_var(
                        l,
                        &mut num_index,
                        &mut numeric_list,
                        &mut numeric_init_vec,
                        &mut instantiated_num_axioms,
                        &mut derived_axiom_index,
                    );
                    let right_name = ensure_expr_var(
                        r,
                        &mut num_index,
                        &mut numeric_list,
                        &mut numeric_init_vec,
                        &mut instantiated_num_axioms,
                        &mut derived_axiom_index,
                    );
                    if let (Some(left_key), Some(right_key)) = (left_name, right_name) {
                        if let (Some(&ni), Some(&nj)) = (num_index.get(&left_key), num_index.get(&right_key)) {
                            let comp_key = format!("{} {} {}", opstr, left_key, right_key);
                            let effect_idx = if let Some(&ei) = var_index.get(&comp_key) {
                                ei
                            } else {
                                let ei = vars.len();
                                var_index.insert(comp_key.clone(), ei);
                                let pos = format!("{} {} {}", opstr, left_key, right_key);
                                let neg = format!("not {} {} {}", opstr, left_key, right_key);
                                vars.push(crate::translate::sas::Variable {
                                    value_names: vec![pos, neg, "<none of those>".to_string()],
                                });
                                ranges.push(3);
                                ei
                            };
                            comp_axioms.push(crate::translate::sas::CompareAxiom {
                                comp: normalize_op(opstr),
                                parts: vec![ni, nj],
                                effect_var: effect_idx,
                            });
                        }
                    }
                }
                _ => {}
            }
        }
        if let Some(eff) = &op.eff {
            match eff {
                crate::translate::pddl_ast::Effect::Add(name, args) => {
                    let atom = format!("{}({})", name, args.join(", "));
                    if let Some(&(v, val)) = atom_to_fdr.get(&atom) {
                        // effects tuple: (var, pre, post, condition)
                        // For add effects, pre = range-1 (none of those) typically
                        let pre = ranges.get(v).map(|r| r - 1).unwrap_or(0);
                        effects.push((v, pre, val, vec![]));
                    }
                }
                crate::translate::pddl_ast::Effect::Del(name, args) => {
                    let atom = format!("{}({})", name, args.join(", "));
                    if let Some(&(v, val)) = atom_to_fdr.get(&atom) {
                        let none_idx = ranges[v] - 1;
                        // pre is the value being deleted, post is none_of_those
                        effects.push((v, val, none_idx, vec![]));
                    }
                }
                crate::translate::pddl_ast::Effect::And(v) => {
                    for sub in v {
                        match sub {
                            crate::translate::pddl_ast::Effect::Add(name, args) => {
                                let atom = format!("{}({})", name, args.join(", "));
                                if let Some(&(v, val)) = atom_to_fdr.get(&atom) {
                                    let pre = ranges.get(v).map(|r| r - 1).unwrap_or(0);
                                    effects.push((v, pre, val, vec![]));
                                }
                            }
                            crate::translate::pddl_ast::Effect::Del(name, args) => {
                                let atom = format!("{}({})", name, args.join(", "));
                                if let Some(&(v, val)) = atom_to_fdr.get(&atom) {
                                    let none_idx = ranges[v] - 1;
                                    effects.push((v, val, none_idx, vec![]));
                                }
                            }
                            crate::translate::pddl_ast::Effect::Increase(nname, _args, val) => {
                                if let Some(&ni) = num_index.get(nname) {
                                    // For now, create a constant numeric variable for the value
                                    let const_name = format!("const:{}", val);
                                    let rhs_idx = if let Some(&idx) = num_index.get(&const_name) {
                                        idx
                                    } else {
                                        let idx = numeric_list.len();
                                        num_index.insert(const_name.clone(), idx);
                                        numeric_list.push(crate::translate::sas::NumericVariable {
                                            name: const_name,
                                            initial: Some(*val),
                                            ntype: "C".to_string(),
                                            axiom_layer: -1,
                                        });
                                        numeric_init_vec.push(*val);
                                        idx
                                    };
                                    op_numeric_effects.push((ni, "+".to_string(), rhs_idx, vec![]));
                                }
                            }
                            crate::translate::pddl_ast::Effect::Decrease(nname, _args, val) => {
                                if let Some(&ni) = num_index.get(nname) {
                                    let const_name = format!("const:{}", val);
                                    let rhs_idx = if let Some(&idx) = num_index.get(&const_name) {
                                        idx
                                    } else {
                                        let idx = numeric_list.len();
                                        num_index.insert(const_name.clone(), idx);
                                        numeric_list.push(crate::translate::sas::NumericVariable {
                                            name: const_name,
                                            initial: Some(*val),
                                            ntype: "C".to_string(),
                                            axiom_layer: -1,
                                        });
                                        numeric_init_vec.push(*val);
                                        idx
                                    };
                                    op_numeric_effects.push((ni, "-".to_string(), rhs_idx, vec![]));
                                }
                            }
                            crate::translate::pddl_ast::Effect::And(_) => { /* nested And - ignore */
                            }
                        }
                    }
                }
                crate::translate::pddl_ast::Effect::Increase(nname, _args, val) => {
                    if let Some(&ni) = num_index.get(nname) {
                        let const_name = format!("const:{}", val);
                        let rhs_idx = if let Some(&idx) = num_index.get(&const_name) {
                            idx
                        } else {
                            let idx = numeric_list.len();
                            num_index.insert(const_name.clone(), idx);
                            numeric_list.push(crate::translate::sas::NumericVariable {
                                name: const_name,
                                initial: Some(*val),
                                ntype: "C".to_string(),
                                axiom_layer: -1,
                            });
                            numeric_init_vec.push(*val);
                            idx
                        };
                        op_numeric_effects.push((ni, "+".to_string(), rhs_idx, vec![]));
                    }
                }
                crate::translate::pddl_ast::Effect::Decrease(nname, _args, val) => {
                    if let Some(&ni) = num_index.get(nname) {
                        let const_name = format!("const:{}", val);
                        let rhs_idx = if let Some(&idx) = num_index.get(&const_name) {
                            idx
                        } else {
                            let idx = numeric_list.len();
                            num_index.insert(const_name.clone(), idx);
                            numeric_list.push(crate::translate::sas::NumericVariable {
                                name: const_name,
                                initial: Some(*val),
                                ntype: "C".to_string(),
                                axiom_layer: -1,
                            });
                            numeric_init_vec.push(*val);
                            idx
                        };
                        op_numeric_effects.push((ni, "-".to_string(), rhs_idx, vec![]));
                    }
                }
            }
        }

        operators.push(crate::translate::sas::SASOperator {
            name: op.name.clone(),
            prevails,
            effects,
            numeric_effects: op_numeric_effects,
            cost: 1.0,
        });
    }
    // Build simple instantiated numeric axioms from numeric init facts we collected.
    // Currently we only support Var = const forms (from problem init). This produces
    // InstantiatedNumericAxiom objects that numeric_axiom_rules can consume to detect
    // constant axioms and compute layers. We'll extend this to full arithmetic
    // expression trees later.
    for (n, init_val) in num_entries.iter() {
        // n is like "fname(arg1,arg2)"
        if let Some(open) = n.find('(') {
            if let Some(close) = n.rfind(')') {
                let fname = n[..open].to_string();
                let args_str = &n[open + 1..close];
                let args: Vec<String> = if args_str.is_empty() {
                    Vec::new()
                } else {
                    args_str.split(',').map(|s| s.to_string()).collect()
                };
                let pne = crate::translate::numeric_axiom_rules::PrimitiveNumericExpression {
                    name: fname.clone(),
                    args: args.clone(),
                };
                let part = crate::translate::numeric_axiom_rules::NumericPart::Constant(
                    crate::translate::numeric_axiom_rules::NumericConstant(*init_val),
                );
                let ax = crate::translate::numeric_axiom_rules::InstantiatedNumericAxiom {
                    name: n.clone(),
                    op: None,
                    parts: vec![part],
                    effect: pne,
                };
                instantiated_num_axioms.push(ax);
            }
        }
    }

    let (num_axioms_by_layer, _max_layer, _num_axiom_map, _const_num_axioms) =
        crate::translate::numeric_axiom_rules::handle_axioms(&instantiated_num_axioms);

    // Propagate computed axiom layers back into numeric_list entries by matching axiom.name
    let mut axname_to_layer: std::collections::HashMap<String, i32> =
        std::collections::HashMap::new();
    for (layer, axs) in &num_axioms_by_layer {
        for ax in axs {
            axname_to_layer.insert(ax.name.clone(), *layer);
        }
    }
    for nv in &mut numeric_list {
        if let Some(l) = axname_to_layer.get(&nv.name) {
            nv.axiom_layer = *l;
        }
    }

    // Build propositional init vector aligned to variables using problem.init
    let mut prop_init: Vec<i32> = vec![-1; vars.len()];
    for sexpr in &prob.init {
        if let crate::translate::pddl_parser::SExpr::List(list) = sexpr {
            if list.is_empty() {
                continue;
            }
            if let crate::translate::pddl_parser::SExpr::Atom(pred) = &list[0] {
                if pred == "=" {
                    continue;
                }
                let args: Vec<String> = list[1..]
                    .iter()
                    .filter_map(|x| match x {
                        crate::translate::pddl_parser::SExpr::Atom(a) => Some(a.clone()),
                        _ => None,
                    })
                    .collect();
                let atom = format!("{}({})", pred, args.join(", "));
                if let Some(&(v, val)) = atom_to_fdr.get(&atom) {
                    prop_init[v] = val as i32;
                }
            }
        }
    }
    for (v_idx, init_val) in prop_init.iter_mut().enumerate() {
        if *init_val == -1 {
            if let Some(idx) = vars[v_idx]
                .value_names
                .iter()
                .position(|name| name == "<none of those>")
            {
                *init_val = idx as i32;
            } else if let Some(idx) = vars[v_idx]
                .value_names
                .iter()
                .position(|name| name.starts_with("NegatedAtom "))
            {
                *init_val = idx as i32;
            } else if ranges.get(v_idx).copied().unwrap_or(0) > 0 {
                *init_val = (ranges[v_idx] as i32) - 1;
            } else {
                *init_val = 0;
            }
        }
    }

    // Build goal pairs from problem.goal, handling comparisons
    let mut goal_pairs: Vec<(usize, usize)> = Vec::new();
    if let Some(g) = &prob.goal {
        let goal_cond = crate::translate::pddl_ast::sexpr_to_condition(g);
        fn handle_goal_cond(
            cond: &crate::translate::pddl_ast::Condition,
            atom_to_fdr: &std::collections::HashMap<String, (usize, usize)>,
            goal_pairs: &mut Vec<(usize, usize)>,
            ensure_expr_var: &mut dyn FnMut(&crate::translate::pddl_parser::SExpr,
                &mut std::collections::HashMap<String, usize>,
                &mut Vec<crate::translate::sas::NumericVariable>,
                &mut Vec<i64>,
                &mut Vec<crate::translate::numeric_axiom_rules::InstantiatedNumericAxiom>,
                &mut std::collections::HashMap<String, usize>) -> Option<String>,
            num_index: &mut std::collections::HashMap<String, usize>,
            numeric_list: &mut Vec<crate::translate::sas::NumericVariable>,
            numeric_init_vec: &mut Vec<i64>,
            instantiated_num_axioms: &mut Vec<crate::translate::numeric_axiom_rules::InstantiatedNumericAxiom>,
            derived_axiom_index: &mut std::collections::HashMap<String, usize>,
            var_index: &mut std::collections::HashMap<String, usize>,
            vars: &mut Vec<crate::translate::sas::Variable>,
            ranges: &mut Vec<usize>,
            comp_axioms: &mut Vec<crate::translate::sas::CompareAxiom>,
            normalize_op: &dyn Fn(&str) -> String,
        ) {
            match cond {
                crate::translate::pddl_ast::Condition::Atom(name, args) => {
                    let atom = format!("{}({})", name, args.join(", "));
                    if let Some(&(v, val)) = atom_to_fdr.get(&atom) {
                        goal_pairs.push((v, val));
                    }
                }
                crate::translate::pddl_ast::Condition::Comparison(opstr, l, r) => {
                    // Use ensure_expr_var on both operands to get their numeric variable names
                    // This handles both PNEs and constants (which become derived!{v}.0())
                    let left_name = ensure_expr_var(
                        l,
                        num_index,
                        numeric_list,
                        numeric_init_vec,
                        instantiated_num_axioms,
                        derived_axiom_index,
                    );
                    let right_name = ensure_expr_var(
                        r,
                        num_index,
                        numeric_list,
                        numeric_init_vec,
                        instantiated_num_axioms,
                        derived_axiom_index,
                    );
                    if let (Some(left_key), Some(right_key)) = (left_name, right_name) {
                        if let (Some(&ni), Some(&nj)) = (num_index.get(&left_key), num_index.get(&right_key)) {
                            let comp_key = format!("{} {} {}", opstr, left_key, right_key);
                            let effect_idx = if let Some(&ei) = var_index.get(&comp_key) {
                                ei
                            } else {
                                let ei = vars.len();
                                var_index.insert(comp_key.clone(), ei);
                                let pos = format!("{} {} {}", opstr, left_key, right_key);
                                let neg = format!("not {} {} {}", opstr, left_key, right_key);
                                vars.push(crate::translate::sas::Variable {
                                    value_names: vec![pos, neg, "<none of those>".to_string()],
                                });
                                ranges.push(3);
                                ei
                            };
                            comp_axioms.push(crate::translate::sas::CompareAxiom {
                                comp: normalize_op(opstr),
                                parts: vec![ni, nj],
                                effect_var: effect_idx,
                            });
                            goal_pairs.push((effect_idx, 0));
                        }
                    }
                }
                crate::translate::pddl_ast::Condition::And(list) => {
                    for c in list {
                        handle_goal_cond(
                            c,
                            atom_to_fdr,
                            goal_pairs,
                            ensure_expr_var,
                            num_index,
                            numeric_list,
                            numeric_init_vec,
                            instantiated_num_axioms,
                            derived_axiom_index,
                            var_index,
                            vars,
                            ranges,
                            comp_axioms,
                            normalize_op,
                        );
                    }
                }
                _ => {}
            }
        }
        handle_goal_cond(
            &goal_cond,
            &atom_to_fdr,
            &mut goal_pairs,
            &mut ensure_expr_var,
            &mut num_index,
            &mut numeric_list,
            &mut numeric_init_vec,
            &mut instantiated_num_axioms,
            &mut derived_axiom_index,
            &mut var_index,
            &mut vars,
            &mut ranges,
            &mut comp_axioms,
            &normalize_op,
        );
    }

    let canonical_variables: Vec<crate::translate::sas::CanonicalVariable> = vars
        .iter()
        .enumerate()
        .map(|(idx, v)| crate::translate::sas::CanonicalVariable {
            name: format!("var{}", idx),
            axiom_layer: -1,
            values: v.value_names.clone(),
        })
        .collect();

    let canonical_operators: Vec<crate::translate::sas::CanonicalOperator> = operators
        .iter()
        .map(|op| {
            let pre_post = op
                .effects
                .iter()
                .map(|(var, pre, post, cond)| crate::translate::sas::CanonicalEffect {
                    var: *var,
                    pre: Some(*pre),
                    post: *post,
                    condition: cond.clone(),
                })
                .collect();
            let assign_effects = op
                .numeric_effects
                .iter()
                .map(|(target, assign_op, rhs_var, cond)| {
                    crate::translate::sas::CanonicalAssignEffect {
                        target: *target,
                        op: assign_op.clone(),
                        rhs: crate::translate::sas::CanonicalAssignRhs::Variable(*rhs_var),
                        condition: cond.clone(),
                    }
                })
                .collect();
            crate::translate::sas::CanonicalOperator {
                name: op.name.clone(),
                prevail: op.prevails.clone(),
                pre_post,
                assign_effects,
                cost: 1.0,
            }
        })
        .collect();

    // We don't yet populate comparison axioms from other sources here; leave empty for now.
    SASTask {
        variables: vars,
        operators,
        numeric_variables: numeric_list,
        numeric_axioms,
        comparison_axioms: comp_axioms,
        axioms: Vec::new(),
        numeric_init: numeric_init_vec.iter().map(|&v| v as f64).collect(),
        mutex_groups: mutex_groups_pairs,
        ranges: ranges.clone(),
        axiom_layers: vec![-1; ranges.len()],
        init: prop_init,
        goal: goal_pairs,
        translation_key: translation_key.clone(),
        canonical_variables,
        canonical_operators,
        canonical_metric: Some(("<".to_string(), metric_idx)),
        metric: ("<".to_string(), metric_idx),
        global_constraint: None,
        comp_axiom_layer: 0,
    }
}
