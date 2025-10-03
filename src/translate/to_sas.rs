use crate::translate::instantiate::GroundedOp;
use crate::translate::pddl_ast::{Problem};
use crate::translate::sas::SASTask;

/// Build boolean variables for each grounded atom occurring in init/pre/effects.
pub fn build_sas(ops: &[GroundedOp], prob: &Problem, external_instantiated_num_axioms: &Vec<crate::translate::numeric_axiom_rules::InstantiatedNumericAxiom>, py_groups: Option<Vec<Vec<String>>>) -> SASTask {
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
    let mut atom_to_fdr: std::collections::HashMap<String, (usize, usize)> = std::collections::HashMap::new();
    let mut vars: Vec<crate::translate::sas::Variable> = Vec::new();
    // helper removed: add_comparison_var was unused
    // collect instantiated numeric axioms discovered while processing expressions
    let mut instantiated_num_axioms: Vec<crate::translate::numeric_axiom_rules::InstantiatedNumericAxiom> = external_instantiated_num_axioms.clone();
    // helper: given an SExpr that may be an arithmetic expression like
    // (+ (f a) (g b)) produce or reuse a numeric variable name, possibly
    // creating an InstantiatedNumericAxiom for the derived expression.
    // Map of derived expression name -> index into instantiated_num_axioms
    let mut derived_axiom_index: std::collections::HashMap<String, usize> = std::collections::HashMap::new();

    // recursive helper: ensure a numeric variable exists for the expression; returns its canonical name
    let ensure_expr_var = |sexpr: &crate::translate::pddl_parser::SExpr,
                               num_index: &mut std::collections::HashMap<String, usize>,
                               numeric_list: &mut Vec<crate::translate::sas::NumericVariable>,
                               numeric_init_vec: &mut Vec<i64>,
                               instantiated_num_axioms: &mut Vec<crate::translate::numeric_axiom_rules::InstantiatedNumericAxiom>,
                               derived_axiom_index: &mut std::collections::HashMap<String, usize>|
                               -> Option<String> {
        // inner recursive function
        fn visit(sexpr: &crate::translate::pddl_parser::SExpr,
                 num_index: &mut std::collections::HashMap<String, usize>,
                 numeric_list: &mut Vec<crate::translate::sas::NumericVariable>,
                 numeric_init_vec: &mut Vec<i64>,
                 instantiated_num_axioms: &mut Vec<crate::translate::numeric_axiom_rules::InstantiatedNumericAxiom>,
                 derived_axiom_index: &mut std::collections::HashMap<String, usize>) -> Option<String> {
        match sexpr {
            crate::translate::pddl_parser::SExpr::List(inner) => {
                if inner.is_empty() { return None; }
                if let crate::translate::pddl_parser::SExpr::Atom(op) = &inner[0] {
                        if op == "+" || op == "-" || op == "*" || op == "/" {
                            if inner.len() != 3 { return None; }
                            // process operands recursively
                            let mut parts_keys: Vec<String> = Vec::new();
                            let mut parts_numericparts: Vec<crate::translate::numeric_axiom_rules::NumericPart> = Vec::new();
                            for operand in &inner[1..] {
                                match operand {
                                    crate::translate::pddl_parser::SExpr::List(_) => {
                                        if let Some(sub_name) = visit(operand, num_index, numeric_list, numeric_init_vec, instantiated_num_axioms, derived_axiom_index) {
                                            let pne_name = sub_name.clone();
                                            let pne = crate::translate::numeric_axiom_rules::PrimitiveNumericExpression { name: pne_name.clone(), args: vec![] };
                                            parts_numericparts.push(crate::translate::numeric_axiom_rules::NumericPart::Primitive(pne));
                                            parts_keys.push(pne_name);
                                        } else { return None; }
                                    }
                                    crate::translate::pddl_parser::SExpr::Atom(atom) => {
                                        if let Ok(nv) = atom.parse::<i64>() {
                                            parts_numericparts.push(crate::translate::numeric_axiom_rules::NumericPart::Constant(crate::translate::numeric_axiom_rules::NumericConstant(nv)));
                                            parts_keys.push(format!("const:{}", nv));
                                        } else {
                                            return None;
                                        }
                                    }
                                }
                            }
                            // determine canonical key for commutative ops
                            let mut key_elements = parts_keys.clone();
                            if op == "+" || op == "*" {
                                key_elements.sort();
                            }
                            let derived_name = format!("({} {})", op, key_elements.join(" "));
                            if !num_index.contains_key(&derived_name) {
                                let idx = numeric_list.len();
                                num_index.insert(derived_name.clone(), idx);
                                numeric_list.push(crate::translate::sas::NumericVariable { name: derived_name.clone(), initial: None });
                                numeric_init_vec.push(0);
                                let effect = crate::translate::numeric_axiom_rules::PrimitiveNumericExpression { name: derived_name.clone(), args: vec![] };
                                let ax = crate::translate::numeric_axiom_rules::InstantiatedNumericAxiom { name: derived_name.clone(), op: Some(op.clone()), parts: parts_numericparts, effect };
                                let ai = instantiated_num_axioms.len();
                                instantiated_num_axioms.push(ax.clone());
                                derived_axiom_index.insert(derived_name.clone(), ai);
                            }
                            return Some(derived_name);
                        }
                        // not an arithmetic op -> treat as PNE (list with fname and args)
                        if let crate::translate::pddl_parser::SExpr::Atom(fname) = &inner[0] {
                            let args = inner[1..].iter().filter_map(|x| match x { crate::translate::pddl_parser::SExpr::Atom(a)=>Some(a.clone()), _=>None }).collect::<Vec<_>>();
                            let key = format!("{}({})", fname, args.join(", "));
                            return Some(key);
                        }
                        None
                    } else { None }
                }
                crate::translate::pddl_parser::SExpr::Atom(a) => {
                    // atom alone is either a constant or unsupported
                    if let Ok(v) = a.parse::<i64>() { Some(format!("const:{}", v)) } else { None }
                }
            }
        }
        visit(sexpr, num_index, numeric_list, numeric_init_vec, instantiated_num_axioms, derived_axiom_index)
    };
    // collect numeric inits and boolean init atoms
    let mut numeric_inits: Vec<(String, i64)> = Vec::new();
    // temporary set of grounded boolean atoms encountered
    let mut grounded_atoms: Vec<String> = Vec::new();
    for a in &prob.init {
        if let crate::translate::pddl_parser::SExpr::List(list) = a {
            if list.len() >= 3 {
                if let crate::translate::pddl_parser::SExpr::Atom(op) = &list[0] {
                    if op == "=" {
                        if let crate::translate::pddl_parser::SExpr::List(left) = &list[1] {
                            if let crate::translate::pddl_parser::SExpr::Atom(fname) = &left[0] {
                                let arg_s = left[1..].iter().filter_map(|x| match x { crate::translate::pddl_parser::SExpr::Atom(s)=>Some(s.clone()), _=>None }).collect::<Vec<_>>().join(", ");
                                let key = format!("{}({})", fname, arg_s);
                                if let crate::translate::pddl_parser::SExpr::Atom(val) = &list[2] {
                                    if let Ok(n) = val.parse::<i64>() {
                                            grounded_atoms.push(key.clone());
                                            numeric_inits.push((key, n));
                                        }
                                }
                            }
                        }
                    } else {
                        // boolean init atom
                        if let crate::translate::pddl_parser::SExpr::Atom(name) = &list[0] {
                                let atom = format!("{}({})", name, list[1..].iter().filter_map(|x| match x { crate::translate::pddl_parser::SExpr::Atom(s)=>Some(s.clone()), _=>None }).collect::<Vec<_>>().join(", "));
                            grounded_atoms.push(atom.clone());
                        }
                    }
                }
            }
        }
    }

    // First pass: collect atoms from ops and numeric var effect hints
    let mut numeric_vars: Vec<(String, i64)> = Vec::new();
    for op in ops {
        if let Some(eff) = &op.eff {
            match eff {
                crate::translate::pddl_ast::Effect::Add(name, args) => { grounded_atoms.push(format!("{}({})", name, args.join(", "))); }
                crate::translate::pddl_ast::Effect::Del(name, args) => { grounded_atoms.push(format!("{}({})", name, args.join(", "))); }
                crate::translate::pddl_ast::Effect::And(v) => {
                            for sub in v {
                        match sub {
                            crate::translate::pddl_ast::Effect::Add(name,args) => { grounded_atoms.push(format!("{}({})", name, args.join(", "))); }
                            crate::translate::pddl_ast::Effect::Del(name,args) => { grounded_atoms.push(format!("{}({})", name, args.join(", "))); }
                            crate::translate::pddl_ast::Effect::Increase(name, _args, val) => { numeric_vars.push((name.clone(), *val)); }
                            crate::translate::pddl_ast::Effect::Decrease(name, _args, val) => { numeric_vars.push((name.clone(), -*val)); }
                            crate::translate::pddl_ast::Effect::And(_) => { /* nested And - ignore for now */ }
                        }
                    }
                }
                crate::translate::pddl_ast::Effect::Increase(name, _args, v) => { numeric_vars.push((name.clone(), *v)); }
                crate::translate::pddl_ast::Effect::Decrease(name, _args, v) => { numeric_vars.push((name.clone(), -*v)); }
            }
        }
        if let Some(pre) = &op.pre {
            match pre {
                crate::translate::pddl_ast::Condition::Atom(name,args) => { grounded_atoms.push(format!("{}({})", name, args.join(", "))); }
                crate::translate::pddl_ast::Condition::And(v) => { for c in v { match c { crate::translate::pddl_ast::Condition::Atom(name,args) => { grounded_atoms.push(format!("{}({})", name, args.join(", "))); }, crate::translate::pddl_ast::Condition::Comparison(_,_,_) => {}, crate::translate::pddl_ast::Condition::Not(_) => {}, crate::translate::pddl_ast::Condition::And(_) => {}, crate::translate::pddl_ast::Condition::True => {}, } } }
                crate::translate::pddl_ast::Condition::Comparison(_, _, _) => { /* handled later */ }
                crate::translate::pddl_ast::Condition::Not(_) => { /* ignore */ }
                crate::translate::pddl_ast::Condition::True => { /* ignore */ }
            }
        }
    }

    // Compute fact groups: prefer externally provided Python groups for faithful semantics,
    // otherwise fall back to the simplified Rust grouping implementation.
    let translation_key: Vec<Vec<String>> = if let Some(pg) = py_groups {
        pg
    } else {
        let (_groups, _mutex_groups, tk) = crate::translate::fact_groups::compute_groups_from_atoms(&grounded_atoms);
        tk
    };
    // Build variables from translation_key
    for (var_no, group_values) in translation_key.iter().enumerate() {
        let mut value_names: Vec<String> = Vec::new();
        // positive atoms first
        for v in group_values {
            value_names.push(format!("Atom {}", v));
        }
        // append negation or <none of those>
        if group_values.len() == 1 {
            // binary variable: add NegatedAtom <atom>
            value_names.push(format!("NegatedAtom {}", group_values[0]));
        } else {
            value_names.push("<none of those>".to_string());
        }
        vars.push(crate::translate::sas::Variable { value_names: value_names.clone() });
        // map each positive atom to (var, val)
        for (val_no, val) in group_values.iter().enumerate() {
            atom_to_fdr.insert(val.clone(), (var_no, val_no));
        }
    }
    // First pass: collect numeric effect hints already done earlier in grounded_atoms collection
    let mut numeric_vars: Vec<(String, i64)> = Vec::new();
    // (numeric_vars were already populated above into numeric_vars by the previous pass)

    // fold numeric vars into NumericVariable structs; prefer init values from problem init, otherwise use effect values
    let mut num_map: std::collections::HashMap<String, i64> = std::collections::HashMap::new();
    for (n, v) in numeric_vars.into_iter() {
        num_map.entry(n).or_insert(v);
    }
    // override with numeric_inits from problem init if present
    for (k, v) in numeric_inits.into_iter() {
        num_map.insert(k, v);
    }

    let mut numeric_list: Vec<crate::translate::sas::NumericVariable> = Vec::new();
    let mut numeric_init_vec: Vec<i64> = Vec::new();
    for (n, v) in num_map.iter() {
        numeric_list.push(crate::translate::sas::NumericVariable { name: n.clone(), initial: Some(*v) });
        numeric_init_vec.push(*v);
    }

    // build mapping from numeric name to index
    let mut num_index: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for (i, nv) in numeric_list.iter().enumerate() {
        num_index.insert(nv.name.clone(), i);
    }

    // second pass: build operators with prevails/effects and numeric_effects
    let mut operators: Vec<crate::translate::sas::SASOperator> = Vec::new();
    let mut numeric_axioms: Vec<crate::translate::sas::NumericAxiom> = Vec::new();
    let mut ax_index: std::collections::HashMap<crate::translate::sas::NumericAxiom, usize> = std::collections::HashMap::new();
    let mut comp_axioms: Vec<crate::translate::sas::CompareAxiom> = Vec::new();
    let _comp_index: std::collections::HashMap<(String, Vec<usize>, usize), usize> = std::collections::HashMap::new();
    for op in ops {
    let mut prevails: Vec<(usize, usize)> = Vec::new();
        let mut effects: Vec<(usize, Option<usize>, usize)> = Vec::new();
        let mut op_numeric_effects: Vec<(usize, i64)> = Vec::new();
    let mut op_numeric_preconds: Vec<crate::translate::sas::NumericPrecond> = Vec::new();

        if let Some(pre) = &op.pre {
            match pre {
                crate::translate::pddl_ast::Condition::Atom(name,args) => {
                    let atom = format!("{}({})", name, args.join(", "));
                    if let Some(&(v, val)) = atom_to_fdr.get(&atom) { prevails.push((v, val)); }
                }
                crate::translate::pddl_ast::Condition::And(v) => {
                    for c in v {
                        match c {
                            crate::translate::pddl_ast::Condition::Atom(name,args) => { let atom = format!("{}({})", name, args.join(", ")); if let Some(&(v,val)) = atom_to_fdr.get(&atom) { prevails.push((v,val)); } }
                            crate::translate::pddl_ast::Condition::Comparison(opstr, l, r) => {
                                    // attempt to parse comparison where left or right is a numeric fluent,
                                    // or an arithmetic expression. We now support simple binary arithmetic
                                    // expressions like (+ (f ...) (g ...)).
                                    let try_parse_sexpr = |s: &crate::translate::pddl_parser::SExpr| -> Option<String> {
                                        match s {
                                            crate::translate::pddl_parser::SExpr::List(inner) if !inner.is_empty() => {
                                                if let crate::translate::pddl_parser::SExpr::Atom(fname) = &inner[0] {
                                                    // treat operator names differently; caller may handle
                                                    if fname == "+" || fname == "-" || fname == "*" || fname == "/" {
                                                        return None;
                                                    }
                                                    let args = inner[1..].iter().filter_map(|x| match x { crate::translate::pddl_parser::SExpr::Atom(a)=>Some(a.clone()), _=>None }).collect::<Vec<_>>().join(",");
                                                    return Some(format!("{}({})", fname, args));
                                                }
                                                None
                                            }
                                            _ => None,
                                        }
                                    };
                                    let parse_int = |s: &crate::translate::pddl_parser::SExpr| -> Option<i64> {
                                        if let crate::translate::pddl_parser::SExpr::Atom(a) = s { a.parse::<i64>().ok() } else { None }
                                    };
                                    // handle arithmetic expressions first
                                    if let Some(derived_left) = ensure_expr_var(l, &mut num_index, &mut numeric_list, &mut numeric_init_vec, &mut instantiated_num_axioms, &mut derived_axiom_index) {
                                        if let Some(fkey2) = try_parse_sexpr(r) {
                                            if let Some(&ni) = num_index.get(&derived_left) {
                                                if let Some(&nj) = num_index.get(&fkey2) {
                                                    op_numeric_preconds.push(crate::translate::sas::NumericPrecond::VarVar(ni, normalize_op(opstr), nj));
                                                    let left_s = derived_left.clone();
                                                    let right_s = fkey2.clone();
                                                    let comp_key = format!("{} {} {}", opstr, left_s, right_s);
                                                    let effect_idx = if let Some(&ei) = var_index.get(&comp_key) { ei } else {
                                                        let ei = vars.len();
                                                        var_index.insert(comp_key.clone(), ei);
                                                        let pos = format!("{} {} {}", opstr, left_s, right_s);
                                                        let neg = format!("not {} {} {}", opstr, left_s, right_s);
                                                        vars.push(crate::translate::sas::Variable { value_names: vec![pos, neg, "<none of those>".to_string()] });
                                                        ei
                                                    };
                                                    comp_axioms.push(crate::translate::sas::CompareAxiom { comp: normalize_op(opstr), parts: vec![ni, nj], effect: effect_idx });
                                                }
                                            }
                                        } else if let Some(vval) = parse_int(r) {
                                            if let Some(&ni) = num_index.get(&derived_left) {
                                                op_numeric_preconds.push(crate::translate::sas::NumericPrecond::VarConst(ni, normalize_op(opstr), vval));
                                            }
                                        }
                                    } else if let Some(derived_right) = ensure_expr_var(r, &mut num_index, &mut numeric_list, &mut numeric_init_vec, &mut instantiated_num_axioms, &mut derived_axiom_index) {
                                        if let Some(vval) = parse_int(l) {
                                            if let Some(&ni) = num_index.get(&derived_right) {
                                                op_numeric_preconds.push(crate::translate::sas::NumericPrecond::VarConst(ni, normalize_op(opstr), vval));
                                            }
                                        }
                                    } else if let Some(fkey) = try_parse_sexpr(l) {
                                        if let Some(vval) = parse_int(r) {
                                            if let Some(&ni) = num_index.get(&fkey) {
                                                op_numeric_preconds.push(crate::translate::sas::NumericPrecond::VarConst(ni, normalize_op(opstr), vval));
                                            }
                                        } else if let Some(fkey2) = try_parse_sexpr(r) {
                                            if let Some(&ni) = num_index.get(&fkey) {
                                                if let Some(&nj) = num_index.get(&fkey2) {
                                                    op_numeric_preconds.push(crate::translate::sas::NumericPrecond::VarVar(ni, normalize_op(opstr), nj));
                                                    // create comparison axiom variable and record CompareAxiom
                                                    let left_s = fkey.clone();
                                                    let right_s = fkey2.clone();
                                                    let comp_key = format!("{} {} {}", opstr, left_s, right_s);
                                                    let effect_idx = if let Some(&ei) = var_index.get(&comp_key) { ei } else {
                                                        let ei = vars.len();
                                                        var_index.insert(comp_key.clone(), ei);
                                                        let pos = format!("{} {} {}", opstr, left_s, right_s);
                                                        let neg = format!("not {} {} {}", opstr, left_s, right_s);
                                                        vars.push(crate::translate::sas::Variable { value_names: vec![pos, neg, "<none of those>".to_string()] });
                                                        ei
                                                    };
                                                    comp_axioms.push(crate::translate::sas::CompareAxiom { comp: normalize_op(opstr), parts: vec![ni, nj], effect: effect_idx });
                                                }
                                            }
                                        }
                                    } else if let Some(fkey) = try_parse_sexpr(r) {
                                        if let Some(vval) = parse_int(l) {
                                            if let Some(&ni) = num_index.get(&fkey) {
                                                op_numeric_preconds.push(crate::translate::sas::NumericPrecond::VarConst(ni, normalize_op(opstr), vval));
                                            }
                                        }
                                    }
                            }
                            crate::translate::pddl_ast::Condition::Not(_) => { /* ignore */ }
                            crate::translate::pddl_ast::Condition::And(_) => { /* ignore */ }
                            crate::translate::pddl_ast::Condition::True => { /* ignore */ }
                        }
                    }
                }
                crate::translate::pddl_ast::Condition::Comparison(opstr, l, r) => {
                    // top-level comparison (operands are SExpr)
                    let try_parse_sexpr = |s: &crate::translate::pddl_parser::SExpr| -> Option<String> {
                        match s {
                            crate::translate::pddl_parser::SExpr::List(inner) if !inner.is_empty() => {
                                if let crate::translate::pddl_parser::SExpr::Atom(fname) = &inner[0] {
                                    let args = inner[1..].iter().filter_map(|x| match x { crate::translate::pddl_parser::SExpr::Atom(a)=>Some(a.clone()), _=>None }).collect::<Vec<_>>().join(",");
                                    return Some(format!("{}({})", fname, args));
                                }
                                None
                            }
                            _ => None,
                        }
                    };
                    let parse_int = |s: &crate::translate::pddl_parser::SExpr| -> Option<i64> { if let crate::translate::pddl_parser::SExpr::Atom(a) = s { a.parse::<i64>().ok() } else { None } };
                    if let Some(fkey) = try_parse_sexpr(l) {
                        if let Some(vval) = parse_int(r) {
                            if let Some(&ni) = num_index.get(&fkey) {
                                op_numeric_preconds.push(crate::translate::sas::NumericPrecond::VarConst(ni, normalize_op(opstr), vval));
                            }
                        } else if let Some(fkey2) = try_parse_sexpr(r) {
                            if let Some(&ni) = num_index.get(&fkey) {
                                if let Some(&nj) = num_index.get(&fkey2) {
                                    op_numeric_preconds.push(crate::translate::sas::NumericPrecond::VarVar(ni, normalize_op(opstr), nj));
                                    // create comparison axiom variable and record CompareAxiom
                                    let left_s = fkey.clone();
                                    let right_s = fkey2.clone();
                                    let comp_key = format!("{} {} {}", opstr, left_s, right_s);
                                    let effect_idx = if let Some(&ei) = var_index.get(&comp_key) { ei } else {
                                        let ei = vars.len();
                                        var_index.insert(comp_key.clone(), ei);
                                        let pos = format!("{} {} {}", opstr, left_s, right_s);
                                        let neg = format!("not {} {} {}", opstr, left_s, right_s);
                                        vars.push(crate::translate::sas::Variable { value_names: vec![pos, neg, "<none of those>".to_string()] });
                                        ei
                                    };
                                    comp_axioms.push(crate::translate::sas::CompareAxiom { comp: normalize_op(opstr), parts: vec![ni, nj], effect: effect_idx });
                                }
                            }
                        }
                    } else if let Some(fkey) = try_parse_sexpr(r) {
                        if let Some(vval) = parse_int(l) {
                            if let Some(&ni) = num_index.get(&fkey) {
                                op_numeric_preconds.push(crate::translate::sas::NumericPrecond::VarConst(ni, normalize_op(opstr), vval));
                            }
                        }
                    }
                }
                _ => {}
            }
        }
        if let Some(eff) = &op.eff {
            match eff {
                crate::translate::pddl_ast::Effect::Add(name,args) => { let atom = format!("{}({})", name, args.join(", ")); if let Some(&(v,val)) = atom_to_fdr.get(&atom) { effects.push((v, None, val)); } }
                crate::translate::pddl_ast::Effect::Del(name,args) => { let atom = format!("{}({})", name, args.join(", ")); if let Some(&(v,val)) = atom_to_fdr.get(&atom) { effects.push((v, Some(val), 0)); } }
                crate::translate::pddl_ast::Effect::And(v) => {
                    for sub in v {
                        match sub {
                            crate::translate::pddl_ast::Effect::Add(name,args) => { let atom = format!("{}({})", name, args.join(", ")); if let Some(&(v,val)) = atom_to_fdr.get(&atom) { effects.push((v, None, val)); } }
                            crate::translate::pddl_ast::Effect::Del(name,args) => { let atom = format!("{}({})", name, args.join(", ")); if let Some(&(v,val)) = atom_to_fdr.get(&atom) { effects.push((v, Some(val), 0)); } }
                            crate::translate::pddl_ast::Effect::Increase(nname, _args, val) => { if let Some(&ni) = num_index.get(nname) { op_numeric_effects.push((ni, *val)); } }
                            crate::translate::pddl_ast::Effect::Decrease(nname, _args, val) => { if let Some(&ni) = num_index.get(nname) { op_numeric_effects.push((ni, -*val)); } }
                            crate::translate::pddl_ast::Effect::And(_) => { /* nested And - ignore */ }
                        }
                    }
                }
                crate::translate::pddl_ast::Effect::Increase(nname, _args, val) => { if let Some(&ni) = num_index.get(nname) { op_numeric_effects.push((ni, *val)); } }
                crate::translate::pddl_ast::Effect::Decrease(nname, _args, val) => { if let Some(&ni) = num_index.get(nname) { op_numeric_effects.push((ni, -*val)); } }
            }
        }

        // map op_numeric_preconds into indices in numeric_axioms and comparison axioms
        let mut precond_idxs: Vec<usize> = Vec::new();
        for np in op_numeric_preconds.into_iter() {
            // convert NumericPrecond -> NumericAxiom
            match np {
                crate::translate::sas::NumericPrecond::VarConst(i, opstr, v) => {
                    let ax = crate::translate::sas::NumericAxiom::VarConst(i, opstr.clone(), v);
                    let idx = if let Some(&existing) = ax_index.get(&ax) { existing } else { let ni = numeric_axioms.len(); numeric_axioms.push(ax.clone()); ax_index.insert(ax.clone(), ni); ni };
                    precond_idxs.push(idx);
                }
                crate::translate::sas::NumericPrecond::VarVar(i, opstr, j) => {
                    // for simple comparisons between numeric variables store as CompareAxiom as well
                    let ax = crate::translate::sas::NumericAxiom::VarVar(i, opstr.clone(), j);
                    let idx = if let Some(&existing) = ax_index.get(&ax) { existing } else { let ni = numeric_axioms.len(); numeric_axioms.push(ax.clone()); ax_index.insert(ax.clone(), ni); ni };
                    precond_idxs.push(idx);
                }
            }
        }

        operators.push(crate::translate::sas::SASOperator { name: op.name.clone(), prevails, effects, numeric_effects: op_numeric_effects, numeric_preconds: precond_idxs });
    }
    // Build simple instantiated numeric axioms from numeric init facts we collected.
    // Currently we only support Var = const forms (from problem init). This produces
    // InstantiatedNumericAxiom objects that numeric_axiom_rules can consume to detect
    // constant axioms and compute layers. We'll extend this to full arithmetic
    // expression trees later.
    for (n, init_val) in num_map.iter() {
        // n is like "fname(arg1,arg2)"
        if let Some(open) = n.find('(') {
            if let Some(close) = n.rfind(')') {
                let fname = n[..open].to_string();
                let args_str = &n[open+1..close];
                let args: Vec<String> = if args_str.is_empty() { Vec::new() } else { args_str.split(',').map(|s| s.to_string()).collect() };
                let pne = crate::translate::numeric_axiom_rules::PrimitiveNumericExpression { name: fname.clone(), args: args.clone() };
                let part = crate::translate::numeric_axiom_rules::NumericPart::Constant(crate::translate::numeric_axiom_rules::NumericConstant(*init_val));
                let ax = crate::translate::numeric_axiom_rules::InstantiatedNumericAxiom { name: n.clone(), op: None, parts: vec![part], effect: pne };
                instantiated_num_axioms.push(ax);
            }
        }
    }

    let (_num_axioms_by_layer, _max_layer, _num_axiom_map, _const_num_axioms) =
        crate::translate::numeric_axiom_rules::handle_axioms(&instantiated_num_axioms);

    // We don't yet populate comparison axioms from other sources here; leave empty for now.
    SASTask { variables: vars, operators, numeric_variables: numeric_list, numeric_axioms, comparison_axioms: comp_axioms, numeric_init: numeric_init_vec }
}
