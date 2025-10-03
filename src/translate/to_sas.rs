use crate::translate::instantiate::GroundedOp;
use crate::translate::pddl_ast::{Problem};
use crate::translate::sas::SASTask;

/// Build boolean variables for each grounded atom occurring in init/pre/effects.
pub fn build_sas(ops: &[GroundedOp], prob: &Problem) -> SASTask {
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
    let mut vars: Vec<crate::translate::sas::Variable> = Vec::new();
    let mut add_atom = |atom: &str| {
        if !var_index.contains_key(atom) {
            let idx = vars.len();
            var_index.insert(atom.to_string(), idx);
            vars.push(crate::translate::sas::Variable { value_names: vec![format!("Atom {}", atom), format!("NegatedAtom {}", atom)] });
        }
    };
    let mut add_comparison_var = |opstr: &str, left_s: &str, right_s: &str, var_index: &mut HashMap<String, usize>, vars: &mut Vec<crate::translate::sas::Variable>| {
        let comp_key = format!("{} {} {}", opstr, left_s, right_s);
        if !var_index.contains_key(&comp_key) {
            let idx = vars.len();
            var_index.insert(comp_key.clone(), idx);
            let pos = format!("{} {} {}", opstr, left_s, right_s);
            let neg = format!("not {} {} {}", opstr, left_s, right_s);
            vars.push(crate::translate::sas::Variable { value_names: vec![pos, neg, "<none of those>".to_string()] });
        }
    };
    // collect numeric inits and boolean init atoms
    let mut numeric_inits: Vec<(String, i64)> = Vec::new();
    for a in &prob.init {
        if let crate::translate::pddl_parser::SExpr::List(list) = a {
            if list.len() >= 3 {
                if let crate::translate::pddl_parser::SExpr::Atom(op) = &list[0] {
                    if op == "=" {
                        if let crate::translate::pddl_parser::SExpr::List(left) = &list[1] {
                            if let crate::translate::pddl_parser::SExpr::Atom(fname) = &left[0] {
                                let arg_s = left[1..].iter().filter_map(|x| match x { crate::translate::pddl_parser::SExpr::Atom(s)=>Some(s.clone()), _=>None }).collect::<Vec<_>>().join(",");
                                let key = format!("{}({})", fname, arg_s);
                                if let crate::translate::pddl_parser::SExpr::Atom(val) = &list[2] {
                                    if let Ok(n) = val.parse::<i64>() {
                                        add_atom(&key);
                                        numeric_inits.push((key, n));
                                    }
                                }
                            }
                        }
                    } else {
                        // boolean init atom
                        if let crate::translate::pddl_parser::SExpr::Atom(name) = &list[0] {
                            let atom = format!("{}({})", name, list[1..].iter().filter_map(|x| match x { crate::translate::pddl_parser::SExpr::Atom(s)=>Some(s.clone()), _=>None }).collect::<Vec<_>>().join(","));
                            add_atom(&atom);
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
                crate::translate::pddl_ast::Effect::Add(name, args) => { add_atom(&format!("{}({})", name, args.join(","))); }
                crate::translate::pddl_ast::Effect::Del(name, args) => { add_atom(&format!("{}({})", name, args.join(","))); }
                crate::translate::pddl_ast::Effect::And(v) => {
                    for sub in v {
                        match sub {
                            crate::translate::pddl_ast::Effect::Add(name,args) => { add_atom(&format!("{}({})", name, args.join(","))); }
                            crate::translate::pddl_ast::Effect::Del(name,args) => { add_atom(&format!("{}({})", name, args.join(","))); }
                            crate::translate::pddl_ast::Effect::Increase(name, _args, val) => { numeric_vars.push((name.clone(), *val)); }
                            crate::translate::pddl_ast::Effect::Decrease(name, _args, val) => { numeric_vars.push((name.clone(), -*val)); }
                            _ => (),
                        }
                    }
                }
                crate::translate::pddl_ast::Effect::Increase(name, _args, v) => { numeric_vars.push((name.clone(), *v)); }
                crate::translate::pddl_ast::Effect::Decrease(name, _args, v) => { numeric_vars.push((name.clone(), -*v)); }
                _ => (),
            }
        }
        if let Some(pre) = &op.pre {
            match pre {
                crate::translate::pddl_ast::Condition::Atom(name,args) => { add_atom(&format!("{}({})", name, args.join(","))); }
                crate::translate::pddl_ast::Condition::And(v) => { for c in v { if let crate::translate::pddl_ast::Condition::Atom(name,args) = c { add_atom(&format!("{}({})", name, args.join(","))); } } }
                _ => (),
            }
        }
    }

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
    let mut comp_index: std::collections::HashMap<(String, Vec<usize>, usize), usize> = std::collections::HashMap::new();
    for op in ops {
    let mut prevails: Vec<(usize, usize)> = Vec::new();
        let mut effects: Vec<(usize, Option<usize>, usize)> = Vec::new();
        let mut op_numeric_effects: Vec<(usize, i64)> = Vec::new();
    let mut op_numeric_preconds: Vec<crate::translate::sas::NumericPrecond> = Vec::new();

        if let Some(pre) = &op.pre {
            match pre {
                crate::translate::pddl_ast::Condition::Atom(name,args) => {
                    let atom = format!("{}({})", name, args.join(","));
                    if let Some(&idx) = var_index.get(&atom) { prevails.push((idx, 1)); }
                }
                crate::translate::pddl_ast::Condition::And(v) => {
                    for c in v {
                        match c {
                            crate::translate::pddl_ast::Condition::Atom(name,args) => { let atom = format!("{}({})", name, args.join(",")); if let Some(&idx) = var_index.get(&atom) { prevails.push((idx,1)); } }
                            crate::translate::pddl_ast::Condition::Comparison(opstr, l, r) => {
                                    // attempt to parse comparison where left or right is a numeric fluent and the other is an integer or fluent
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
                                    let parse_int = |s: &crate::translate::pddl_parser::SExpr| -> Option<i64> {
                                        if let crate::translate::pddl_parser::SExpr::Atom(a) = s { a.parse::<i64>().ok() } else { None }
                                    };
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
                crate::translate::pddl_ast::Effect::Add(name,args) => { let atom = format!("{}({})", name, args.join(",")); if let Some(&idx) = var_index.get(&atom) { effects.push((idx, None, 1)); } }
                crate::translate::pddl_ast::Effect::Del(name,args) => { let atom = format!("{}({})", name, args.join(",")); if let Some(&idx) = var_index.get(&atom) { effects.push((idx, Some(1), 0)); } }
                crate::translate::pddl_ast::Effect::And(v) => {
                    for sub in v {
                        match sub {
                            crate::translate::pddl_ast::Effect::Add(name,args) => { let atom = format!("{}({})", name, args.join(",")); if let Some(&idx) = var_index.get(&atom) { effects.push((idx, None, 1)); } }
                            crate::translate::pddl_ast::Effect::Del(name,args) => { let atom = format!("{}({})", name, args.join(",")); if let Some(&idx) = var_index.get(&atom) { effects.push((idx, Some(1), 0)); } }
                            crate::translate::pddl_ast::Effect::Increase(nname, _args, val) => { if let Some(&ni) = num_index.get(nname) { op_numeric_effects.push((ni, *val)); } }
                            crate::translate::pddl_ast::Effect::Decrease(nname, _args, val) => { if let Some(&ni) = num_index.get(nname) { op_numeric_effects.push((ni, -*val)); } }
                            _ => {}
                        }
                    }
                }
                crate::translate::pddl_ast::Effect::Increase(nname, _args, val) => { if let Some(&ni) = num_index.get(nname) { op_numeric_effects.push((ni, *val)); } }
                crate::translate::pddl_ast::Effect::Decrease(nname, _args, val) => { if let Some(&ni) = num_index.get(nname) { op_numeric_effects.push((ni, -*val)); } }
                _ => {}
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
    let mut instantiated_num_axioms: Vec<crate::translate::numeric_axiom_rules::InstantiatedNumericAxiom> = Vec::new();
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

    let (num_axioms_by_layer, max_layer, num_axiom_map, const_num_axioms) =
        crate::translate::numeric_axiom_rules::handle_axioms(&instantiated_num_axioms);

    // We don't yet populate comparison axioms from other sources here; leave empty for now.
    SASTask { variables: vars, operators, numeric_variables: numeric_list, numeric_axioms, comparison_axioms: comp_axioms, numeric_init: numeric_init_vec }
}
