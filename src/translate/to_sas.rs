use crate::translate::instantiate::GroundedOp;
use crate::translate::pddl_ast::{Domain, Problem};
use crate::translate::sas::SASTask;

/// Format a PrimitiveNumericExpression as a string for use as a numeric variable name.
/// Matches Python's PNE formatting: "name(args)" or just "name()" if no args.
fn format_pne(pne: &crate::translate::numeric_axiom_rules::PrimitiveNumericExpression) -> String {
    if pne.args.is_empty() {
        if pne.name.ends_with(')') {
            pne.name.clone()
        } else {
            format!("{}()", pne.name)
        }
    } else {
        format!("{}({})", pne.name, pne.args.join(", "))
    }
}

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
                // Non-numeric atom - could be a zero-argument function like 'derived!0' or 'funds'
                // Try to look it up as a zero-argument function
                let key = format!("{}()", a);
                if num_index.contains_key(&key) {
                    Some(key)
                } else {
                    // Not found with parentheses, maybe it's already in the right format
                    if num_index.contains_key(a) {
                        Some(a.clone())
                    } else {
                        None
                    }
                }
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
    propositional_axioms: &[crate::translate::normalize::TaskAxiom],
    normalized_goal: &crate::translate::pddl_ast::Condition,
) -> Result<SASTask, String> {
    fn normalize_op(op: &str) -> String {
        // Keep the operator as-is for Python compatibility
        op.to_string()
    }
    fn negate_op(op: &str) -> String {
        match op {
            "<=" => ">".to_string(),
            ">=" => "<".to_string(),
            "<" => ">=".to_string(),
            ">" => "<=".to_string(),
            "=" => "!=".to_string(),
            "!=" => "=".to_string(),
            _ => format!("not {}", op),
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
    // Prune numeric axioms to those required by comparisons and numeric effects
    {
        fn sexpr_to_key(sexpr: &crate::translate::pddl_parser::SExpr) -> Option<String> {
            match sexpr {
                crate::translate::pddl_parser::SExpr::Atom(a) => {
                    if let Ok(v) = a.parse::<i64>() {
                        Some(format!("derived!{}.0()", v))
                    } else if a.ends_with(')') {
                        Some(a.clone())
                    } else {
                        Some(format!("{}()", a))
                    }
                }
                crate::translate::pddl_parser::SExpr::List(list) => {
                    if list.is_empty() {
                        return None;
                    }
                    let fname = match &list[0] {
                        crate::translate::pddl_parser::SExpr::Atom(name) => name.clone(),
                        _ => return None,
                    };
                    let args = list[1..]
                        .iter()
                        .filter_map(|x| match x {
                            crate::translate::pddl_parser::SExpr::Atom(a) => Some(a.clone()),
                            _ => None,
                        })
                        .collect::<Vec<_>>();
                    if args.is_empty() {
                        if fname.ends_with(')') {
                            Some(fname)
                        } else {
                            Some(format!("{}()", fname))
                        }
                    } else {
                        Some(format!("{}({})", fname, args.join(", ")))
                    }
                }
            }
        }

        fn collect_comparisons(
            cond: &crate::translate::pddl_ast::Condition,
            required: &mut std::collections::HashSet<String>,
        ) {
            match cond {
                crate::translate::pddl_ast::Condition::Comparison(_, l, r) => {
                    if let Some(k) = sexpr_to_key(l) {
                        required.insert(k);
                    }
                    if let Some(k) = sexpr_to_key(r) {
                        required.insert(k);
                    }
                }
                crate::translate::pddl_ast::Condition::And(v) => {
                    for c in v {
                        collect_comparisons(c, required);
                    }
                }
                _ => {}
            }
        }

        let mut required_pnes: std::collections::HashSet<String> =
            std::collections::HashSet::new();
        for op in ops {
            if let Some(pre) = &op.pre {
                collect_comparisons(pre, &mut required_pnes);
            }
            if let Some(eff) = &op.eff {
                if let crate::translate::pddl_ast::Effect::Increase(name, args, v)
                | crate::translate::pddl_ast::Effect::Decrease(name, args, v) = eff
                {
                    required_pnes.insert(format!("{}({})", name, args.join(", ")));
                    required_pnes.insert(format!("derived!{}.0()", v));
                }
            }
            for (_conds, eff) in &op.effects {
                match eff {
                    crate::translate::pddl_ast::Effect::Increase(name, args, v)
                    | crate::translate::pddl_ast::Effect::Decrease(name, args, v) => {
                        required_pnes.insert(format!("{}({})", name, args.join(", ")));
                        required_pnes.insert(format!("derived!{}.0()", v));
                    }
                    _ => {}
                }
            }
        }

        let mut needed = required_pnes.clone();
        let mut kept: Vec<crate::translate::numeric_axiom_rules::InstantiatedNumericAxiom> =
            Vec::new();
        let mut changed = true;
        while changed {
            changed = false;
            for ax in &instantiated_num_axioms {
                let effect_key = format_pne(&ax.effect);
                if !needed.contains(&effect_key) {
                    continue;
                }
                if kept.iter().any(|k| k == ax) {
                    continue;
                }
                kept.push(ax.clone());
                for part in &ax.parts {
                    match part {
                        crate::translate::numeric_axiom_rules::NumericPart::Primitive(pne) => {
                            let key = format_pne(pne);
                            if needed.insert(key) {
                                changed = true;
                            }
                        }
                        crate::translate::numeric_axiom_rules::NumericPart::Axiom(ref_ax) => {
                            let key = format_pne(&ref_ax.effect);
                            if needed.insert(key) {
                                changed = true;
                            }
                        }
                        crate::translate::numeric_axiom_rules::NumericPart::Constant(_) => {}
                    }
                }
            }
        }
        instantiated_num_axioms = kept;
    }
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
                            if fluent_preds.contains(name.as_str()) {
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
    let _canon_num = |name: &str| -> String {
        if name.contains('(') {
            name.to_string()
        } else {
            format!("{}()", name)
        }
    };
    for op in ops {
        if !op.effects.is_empty() {
            for (_conds, eff) in &op.effects {
                match eff {
                    crate::translate::pddl_ast::Effect::Add(name, args) => {
                        grounded_atoms.push(format!("{}({})", name, args.join(", ")));
                    }
                    crate::translate::pddl_ast::Effect::Del(name, args) => {
                        grounded_atoms.push(format!("{}({})", name, args.join(", ")));
                    }
                    crate::translate::pddl_ast::Effect::Increase(name, args, val) => {
                        let key = format!("{}({})", name, args.join(", "));
                        numeric_vars.push((key, *val));
                    }
                    crate::translate::pddl_ast::Effect::Decrease(name, args, val) => {
                        let key = format!("{}({})", name, args.join(", "));
                        numeric_vars.push((key, -*val));
                    }
                    crate::translate::pddl_ast::Effect::And(_) => {}
                }
            }
        } else if let Some(eff) = &op.eff {
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
                            crate::translate::pddl_ast::Effect::Increase(name, args, val) => {
                                let key = format!("{}({})", name, args.join(", "));
                                numeric_vars.push((key, *val));
                            }
                            crate::translate::pddl_ast::Effect::Decrease(name, args, val) => {
                                let key = format!("{}({})", name, args.join(", "));
                                numeric_vars.push((key, -*val));
                            }
                            crate::translate::pddl_ast::Effect::And(_) => { /* nested And - ignore for now */
                            }
                        }
                    }
                }
                crate::translate::pddl_ast::Effect::Increase(name, args, v) => {
                    let key = format!("{}({})", name, args.join(", "));
                    numeric_vars.push((key, *v));
                }
                crate::translate::pddl_ast::Effect::Decrease(name, args, v) => {
                    let key = format!("{}({})", name, args.join(", "));
                    numeric_vars.push((key, -*v));
                }
            }
        }
        if let Some(pre) = &op.pre {
            match pre {
                crate::translate::pddl_ast::Condition::Atom(name, args) => {
                    if fluent_preds.contains(name.as_str()) {
                        grounded_atoms.push(format!("{}({})", name, args.join(", ")));
                    }
                }
                crate::translate::pddl_ast::Condition::And(v) => {
                    for c in v {
                        match c {
                            crate::translate::pddl_ast::Condition::Atom(name, args) => {
                                if fluent_preds.contains(name.as_str()) {
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

    // Create propositional variables for axioms (Python uses new-axiom@N naming)
    let mut axiom_atom_to_var: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let mut axiom_name_map: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    let _axiom_layer = 31; // Standard layer for propositional axioms (after comparisons)
    for (idx, ax) in propositional_axioms.iter().enumerate() {
        let ax_name = format!("new-axiom@{}", idx);
        axiom_name_map.insert(ax.name.clone(), ax_name.clone());
        let atom = format!("{}()", ax_name);
        let var_idx = vars.len();
        axiom_atom_to_var.insert(atom.clone(), var_idx);
        vars.push(crate::translate::sas::Variable {
            value_names: vec![
                format!("Atom {}", atom),
                format!("NegatedAtom {}", atom),
            ],
        });
        ranges.push(2); // Derived variables always have range 2
    }

    // Add a dedicated goal axiom variable (Python uses the next new-axiom@N slot)
    let goal_axiom_name = format!("new-axiom@{}", propositional_axioms.len());
    let goal_axiom_atom = format!("{}()", goal_axiom_name);
    let goal_axiom_var = vars.len();
    axiom_atom_to_var.insert(goal_axiom_atom.clone(), goal_axiom_var);
    vars.push(crate::translate::sas::Variable {
        value_names: vec![
            format!("Atom {}", goal_axiom_atom),
            format!("NegatedAtom {}", goal_axiom_atom),
        ],
    });
    ranges.push(2);

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

    // Following Python's strips_to_sas_dictionary logic:
    // 1. First add numeric variables for each instantiated numeric axiom's effect
    // 2. Then add remaining fluents from numeric_inits that aren't already present
    
    let mut numeric_list: Vec<crate::translate::sas::NumericVariable> = Vec::new();
    let mut numeric_init_vec: Vec<i64> = Vec::new();
    let mut num_index: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    
    // Process axiom_rules to get ntype and axiom_layer for each axiom
    let axiom_by_pne = crate::translate::numeric_axiom_rules::axiom_by_pne(&instantiated_num_axioms);
    let constant_axioms = crate::translate::numeric_axiom_rules::identify_constants(
        &instantiated_num_axioms,
        &axiom_by_pne,
    );
    let constant_effects: std::collections::HashSet<_> = constant_axioms.iter().map(|a| &a.effect).collect();
    let (axioms_by_layer, _max_layer) = crate::translate::numeric_axiom_rules::compute_axiom_layers(
        &instantiated_num_axioms,
        &constant_axioms,
        &axiom_by_pne,
    );
    
    // Build axiom_map to identify redundant axioms
    let axiom_map = crate::translate::numeric_axiom_rules::identify_equivalent_axioms(
        &axioms_by_layer,
        &axiom_by_pne,
    );
    
    // First: add numeric variables for each axiom effect (matching Python's order)
    // Python adds axioms first, then fluents, sorted by name
    let mut axiom_effects_added: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut redundant_axioms: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    
    // Sort axioms by effect name for deterministic ordering
    let mut sorted_axioms: Vec<_> = instantiated_num_axioms.iter().collect();
    sorted_axioms.sort_by(|a, b| {
        let a_name = format_pne(&a.effect);
        let b_name = format_pne(&b.effect);
        a_name.cmp(&b_name)
    });
    
    for axiom in sorted_axioms.iter() {
        let effect_name = format_pne(&axiom.effect);
        // Check if this axiom effect is mapped to another (redundant)
        if let Some(mapped_axiom) = axiom_map.get(*axiom) {
            let mapped_name = format_pne(&mapped_axiom.effect);
            redundant_axioms.insert(effect_name.clone(), mapped_name);
            continue;
        }
        
        if !axiom_effects_added.contains(&effect_name) {
            axiom_effects_added.insert(effect_name.clone());
            
            // Determine ntype: 'C' for constants, 'D' for derived
            let ntype = if constant_effects.contains(&axiom.effect) {
                "C".to_string()
            } else {
                "D".to_string()
            };
            
            // Determine axiom_layer
            let mut axiom_layer: i32 = -1;
            for (layer, layer_axioms) in &axioms_by_layer {
                if layer_axioms.iter().any(|a| format_pne(&a.effect) == effect_name) {
                    axiom_layer = *layer;
                    break;
                }
            }
            
            let idx = numeric_list.len();
            num_index.insert(effect_name.clone(), idx);
            numeric_list.push(crate::translate::sas::NumericVariable {
                name: effect_name,
                initial: Some(0),  // Will be updated for constants below
                ntype,
                axiom_layer,
            });
            numeric_init_vec.push(0);
        }
    }
    
    // Handle redundant axiom mappings: map them to the same index as their equivalent
    for (redundant_name, target_name) in &redundant_axioms {
        if let Some(&target_idx) = num_index.get(target_name) {
            num_index.insert(redundant_name.clone(), target_idx);
        }
    }

    // Ensure numeric variables exist for all numeric constants referenced in axioms
    for ax in &instantiated_num_axioms {
        for part in &ax.parts {
            if let crate::translate::numeric_axiom_rules::NumericPart::Constant(c) = part {
                let const_name = format!("derived!{}.0()", c.0);
                if !num_index.contains_key(&const_name) {
                    let idx = numeric_list.len();
                    num_index.insert(const_name.clone(), idx);
                    numeric_list.push(crate::translate::sas::NumericVariable {
                        name: const_name.clone(),
                        initial: Some(c.0),
                        ntype: "C".to_string(),
                        axiom_layer: -1,
                    });
                    numeric_init_vec.push(c.0);
                }
            }
        }
    }
    
    // Now add remaining fluents from numeric_inits that aren't already in the map
    // Combine numeric_vars and numeric_inits, preferring init values
    let mut num_map: std::collections::HashMap<String, i64> = std::collections::HashMap::new();
    let fluent_numeric_keys: std::collections::HashSet<String> =
        numeric_vars.iter().map(|(k, _)| k.clone()).collect();
    for (n, v) in numeric_vars.into_iter() {
        num_map.entry(n).or_insert(v);
    }
    for (k, v) in numeric_inits.into_iter() {
        if fluent_numeric_keys.contains(&k) {
            num_map.insert(k, v);
        }
    }
    // Ensure total-cost() exists for metric
    num_map.entry("total-cost()".to_string()).or_insert(0);
    
    // Sort fluents by name and add those not already from axioms
    let mut fluent_entries: Vec<(String, i64)> =
        num_map.iter().map(|(k, v)| (k.clone(), *v)).collect();
    fluent_entries.sort_by(|a, b| a.0.cmp(&b.0));
    for (n, v) in fluent_entries.iter() {
        // Skip if already added from axioms
        if num_index.contains_key(n) {
            continue;
        }
        
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
        let idx = numeric_list.len();
        num_index.insert(n.clone(), idx);
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

    // num_index is already built above, no need to rebuild

    // second pass: build operators with prevails/effects and numeric_effects
    let mut operators: Vec<crate::translate::sas::SASOperator> = Vec::new();
    let mut numeric_axioms: Vec<crate::translate::sas::NumericAxiom> = Vec::new();
    let mut comp_axioms: Vec<crate::translate::sas::CompareAxiom> = Vec::new();
    // Track which comparison axioms have been added (by effect_var) to avoid duplicates
    let mut comp_axiom_added: std::collections::HashSet<usize> = std::collections::HashSet::new();
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
                                        &l,
                                    &mut num_index,
                                    &mut numeric_list,
                                    &mut numeric_init_vec,
                                    &mut instantiated_num_axioms,
                                    &mut derived_axiom_index,
                                );
                                    let right_name = ensure_expr_var(
                                        &r,
                                    &mut num_index,
                                    &mut numeric_list,
                                    &mut numeric_init_vec,
                                    &mut instantiated_num_axioms,
                                    &mut derived_axiom_index,
                                );
                                if let (Some(left_key), Some(right_key)) = (left_name, right_name) {
                                    if let (Some(&ni), Some(&nj)) = (num_index.get(&left_key), num_index.get(&right_key)) {
                                        let op_norm = normalize_op(&opstr);
                                        let comp_key = format!("{} {} {}", op_norm, ni, nj);
                                        let effect_idx = if let Some(&ei) = var_index.get(&comp_key) {
                                            ei
                                        } else {
                                            let ei = vars.len();
                                            var_index.insert(comp_key.clone(), ei);
                                            let pos = format!("{} {} {}", op_norm, ni, nj);
                                            let neg = format!("{} {} {}", negate_op(&op_norm), ni, nj);
                                            vars.push(crate::translate::sas::Variable {
                                                value_names: vec![pos, neg, "<none of those>".to_string()],
                                            });
                                            ranges.push(3);
                                            ei
                                        };
                                        // Only add the comparison axiom if we haven't seen this effect_var yet
                                        if !comp_axiom_added.contains(&effect_idx) {
                                            comp_axiom_added.insert(effect_idx);
                                            comp_axioms.push(crate::translate::sas::CompareAxiom {
                                                comp: normalize_op(&opstr),
                                                parts: vec![ni, nj],
                                                effect_var: effect_idx,
                                            });
                                        }
                                        prevails.push((effect_idx, 0));
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
                        &l,
                        &mut num_index,
                        &mut numeric_list,
                        &mut numeric_init_vec,
                        &mut instantiated_num_axioms,
                        &mut derived_axiom_index,
                    );
                    let right_name = ensure_expr_var(
                        &r,
                        &mut num_index,
                        &mut numeric_list,
                        &mut numeric_init_vec,
                        &mut instantiated_num_axioms,
                        &mut derived_axiom_index,
                    );
                    if let (Some(left_key), Some(right_key)) = (left_name, right_name) {
                        if let (Some(&ni), Some(&nj)) = (num_index.get(&left_key), num_index.get(&right_key)) {
                            let op_norm = normalize_op(&opstr);
                            let comp_key = format!("{} {} {}", op_norm, ni, nj);
                            let effect_idx = if let Some(&ei) = var_index.get(&comp_key) {
                                ei
                            } else {
                                let ei = vars.len();
                                var_index.insert(comp_key.clone(), ei);
                                let pos = format!("{} {} {}", op_norm, ni, nj);
                                let neg = format!("{} {} {}", negate_op(&op_norm), ni, nj);
                                vars.push(crate::translate::sas::Variable {
                                    value_names: vec![pos, neg, "<none of those>".to_string()],
                                });
                                ranges.push(3);
                                ei
                            };
                            // Only add the comparison axiom if we haven't seen this effect_var yet
                            if !comp_axiom_added.contains(&effect_idx) {
                                comp_axiom_added.insert(effect_idx);
                                comp_axioms.push(crate::translate::sas::CompareAxiom {
                                    comp: normalize_op(&opstr),
                                    parts: vec![ni, nj],
                                    effect_var: effect_idx,
                                });
                            }
                            prevails.push((effect_idx, 0));
                        }
                    }
                }
                _ => {}
            }
        }
        if !op.effects.is_empty() {
            for (conds, eff) in &op.effects {
                let condition = {
                    let mut build_condition_vals = |conds: &[crate::translate::pddl_ast::Condition]| {
                        let mut cond_vals: Vec<(usize, usize)> = Vec::new();
                        let mut stack: Vec<crate::translate::pddl_ast::Condition> = conds.to_vec();
                        while let Some(cond) = stack.pop() {
                            match cond {
                                crate::translate::pddl_ast::Condition::Atom(name, args) => {
                                    let atom = format!("{}({})", name, args.join(", "));
                                    if let Some(&(v, val)) = atom_to_fdr.get(&atom) {
                                        cond_vals.push((v, val));
                                    }
                                }
                                crate::translate::pddl_ast::Condition::Not(inner) => {
                                    if let crate::translate::pddl_ast::Condition::Atom(name, args) = *inner {
                                        let atom = format!("{}({})", name, args.join(", "));
                                        let neg = format!("NegatedAtom {}", atom);
                                        if let Some(&(v, val)) = atom_to_fdr.get(&neg) {
                                            cond_vals.push((v, val));
                                        }
                                    }
                                }
                                crate::translate::pddl_ast::Condition::And(parts) => {
                                    for part in parts {
                                        stack.push(part);
                                    }
                                }
                                crate::translate::pddl_ast::Condition::Comparison(opstr, l, r) => {
                                    let left_name = ensure_expr_var(
                                        &l,
                                        &mut num_index,
                                        &mut numeric_list,
                                        &mut numeric_init_vec,
                                        &mut instantiated_num_axioms,
                                        &mut derived_axiom_index,
                                    );
                                    let right_name = ensure_expr_var(
                                        &r,
                                        &mut num_index,
                                        &mut numeric_list,
                                        &mut numeric_init_vec,
                                        &mut instantiated_num_axioms,
                                        &mut derived_axiom_index,
                                    );
                                    if let (Some(left_key), Some(right_key)) = (left_name, right_name) {
                                        if let (Some(&ni), Some(&nj)) =
                                            (num_index.get(&left_key), num_index.get(&right_key))
                                        {
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
                                            if !comp_axiom_added.contains(&effect_idx) {
                                                comp_axiom_added.insert(effect_idx);
                                                comp_axioms.push(crate::translate::sas::CompareAxiom {
                                                    comp: normalize_op(&opstr),
                                                    parts: vec![ni, nj],
                                                    effect_var: effect_idx,
                                                });
                                            }
                                            cond_vals.push((effect_idx, 0));
                                        }
                                    }
                                }
                                _ => {}
                            }
                        }
                        cond_vals
                    };
                    build_condition_vals(conds)
                };
                match eff {
                    crate::translate::pddl_ast::Effect::Add(name, args) => {
                        let atom = format!("{}({})", name, args.join(", "));
                        if let Some(&(v, val)) = atom_to_fdr.get(&atom) {
                            let pre = ranges.get(v).map(|r| r - 1).unwrap_or(0);
                            effects.push((v, pre, val, condition.clone()));
                        }
                    }
                    crate::translate::pddl_ast::Effect::Del(name, args) => {
                        let atom = format!("{}({})", name, args.join(", "));
                        if let Some(&(v, val)) = atom_to_fdr.get(&atom) {
                            let none_idx = ranges[v] - 1;
                            effects.push((v, val, none_idx, condition.clone()));
                        }
                    }
                    crate::translate::pddl_ast::Effect::Increase(nname, args, val) => {
                        let nkey = format!("{}({})", nname, args.join(", "));
                        if let Some(&ni) = num_index.get(&nkey) {
                            let const_name = format!("derived!{}.0()", val);
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
                            op_numeric_effects.push((ni, "+".to_string(), rhs_idx, condition.clone()));
                        }
                    }
                    crate::translate::pddl_ast::Effect::Decrease(nname, args, val) => {
                        let nkey = format!("{}({})", nname, args.join(", "));
                        if let Some(&ni) = num_index.get(&nkey) {
                            let const_name = format!("derived!{}.0()", val);
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
                            op_numeric_effects.push((ni, "-".to_string(), rhs_idx, condition.clone()));
                        }
                    }
                    crate::translate::pddl_ast::Effect::And(_) => {}
                }
            }
        } else if let Some(eff) = &op.eff {
            match eff {
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
                            crate::translate::pddl_ast::Effect::Increase(nname, args, val) => {
                                let nkey = format!("{}({})", nname, args.join(", "));
                                if let Some(&ni) = num_index.get(&nkey) {
                                    let const_name = format!("derived!{}.0()", val);
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
                            crate::translate::pddl_ast::Effect::Decrease(nname, args, val) => {
                                let nkey = format!("{}({})", nname, args.join(", "));
                                if let Some(&ni) = num_index.get(&nkey) {
                                    let const_name = format!("derived!{}.0()", val);
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
                            crate::translate::pddl_ast::Effect::And(_) => {}
                        }
                    }
                }
                crate::translate::pddl_ast::Effect::Increase(nname, args, val) => {
                    let nkey = format!("{}({})", nname, args.join(", "));
                    if let Some(&ni) = num_index.get(&nkey) {
                        let const_name = format!("derived!{}.0()", val);
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
                crate::translate::pddl_ast::Effect::Decrease(nname, args, val) => {
                    let nkey = format!("{}({})", nname, args.join(", "));
                    if let Some(&ni) = num_index.get(&nkey) {
                        let const_name = format!("derived!{}.0()", val);
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
    
    // Note: We used to build axioms from numeric init facts here, but now we receive
    // the instantiated numeric axioms from the model-guided grounding process.
    // The axiom layer computation is already done above when populating numeric_list.

    let (num_axioms_by_layer, _max_layer, _num_axiom_map, _const_num_axioms) =
        crate::translate::numeric_axiom_rules::handle_axioms_checked(&instantiated_num_axioms)
            .map_err(|err| format!("numeric axiom error: {}", err))?;

    // Evaluate numeric axioms to populate initial values for derived variables
    {
        let pne_key = |name: &str, args: &[String]| -> String {
            if args.is_empty() {
                if name.ends_with(')') {
                    name.to_string()
                } else {
                    format!("{}()", name)
                }
            } else {
                format!("{}({})", name, args.join(", "))
            }
        };
        let mut layers: Vec<i32> = num_axioms_by_layer.keys().copied().collect();
        layers.sort();
        for layer in layers {
            if let Some(axs) = num_axioms_by_layer.get(&layer) {
                for ax in axs {
                    let effect_key = pne_key(&ax.effect.name, &ax.effect.args);
                    let effect_idx = match num_index.get(&effect_key) {
                        Some(idx) => *idx,
                        None => continue,
                    };
                    let mut values: Vec<i64> = Vec::new();
                    let mut ok = true;
                    for part in &ax.parts {
                        match part {
                            crate::translate::numeric_axiom_rules::NumericPart::Constant(c) => {
                                values.push(c.0);
                            }
                            crate::translate::numeric_axiom_rules::NumericPart::Primitive(pne) => {
                                let key = pne_key(&pne.name, &pne.args);
                                if let Some(idx) = num_index.get(&key) {
                                    values.push(numeric_init_vec.get(*idx).copied().unwrap_or(0));
                                } else {
                                    ok = false;
                                    break;
                                }
                            }
                            crate::translate::numeric_axiom_rules::NumericPart::Axiom(ref_ax) => {
                                let key = pne_key(&ref_ax.effect.name, &ref_ax.effect.args);
                                if let Some(idx) = num_index.get(&key) {
                                    values.push(numeric_init_vec.get(*idx).copied().unwrap_or(0));
                                } else {
                                    ok = false;
                                    break;
                                }
                            }
                        }
                    }
                    if !ok || values.is_empty() {
                        continue;
                    }
                    let value = match ax.op.as_deref() {
                        None => values[0],
                        Some("+") => values.into_iter().fold(0, |acc, v| acc + v),
                        Some("-") => {
                            let mut iter = values.into_iter();
                            if let Some(mut acc) = iter.next() {
                                if ax.parts.len() == 1 {
                                    acc = -acc;
                                }
                                for v in iter {
                                    acc -= v;
                                }
                                acc
                            } else {
                                continue;
                            }
                        }
                        Some("*") => values.into_iter().fold(1, |acc, v| acc * v),
                        Some("/") => {
                            let mut iter = values.into_iter();
                            if let Some(mut acc) = iter.next() {
                                for v in iter {
                                    if v == 0 {
                                        ok = false;
                                        break;
                                    }
                                    acc /= v;
                                }
                                if ok { acc } else { continue }
                            } else {
                                continue;
                            }
                        }
                        Some(_) => continue,
                    };
                    if effect_idx < numeric_init_vec.len() {
                        numeric_init_vec[effect_idx] = value;
                        if let Some(nv) = numeric_list.get_mut(effect_idx) {
                            nv.initial = Some(value);
                        }
                    }
                }
            }
        }
    }

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

    // Convert instantiated numeric axioms to SAS NumericAxiom format
    // We need to map the axiom's effect and parts to numeric variable indices
    let format_pne = |name: &str, args: &[String]| -> String {
        if args.is_empty() {
            if name.ends_with(')') {
                name.to_string()
            } else {
                format!("{}()", name)
            }
        } else {
            format!("{}({})", name, args.join(", "))
        }
    };
    for (_layer, axs) in &num_axioms_by_layer {
        for ax in axs {
            // Get the effect's numeric variable index
            let effect_key = format_pne(&ax.effect.name, &ax.effect.args);
            let effect_idx = *num_index
                .get(&effect_key)
                .ok_or_else(|| format!("numeric axiom effect not found: {}", effect_key))?;

            // Get the parts' numeric variable indices
            let mut part_indices: Vec<usize> = Vec::new();
            for part in &ax.parts {
                let part_key = match part {
                    crate::translate::numeric_axiom_rules::NumericPart::Primitive(pne) => {
                        format_pne(&pne.name, &pne.args)
                    }
                    crate::translate::numeric_axiom_rules::NumericPart::Constant(c) => {
                        // Constants are represented as derived!{value}.0()
                        format!("derived!{}.0()", c.0)
                    }
                    crate::translate::numeric_axiom_rules::NumericPart::Axiom(ref_ax) => {
                        // Reference to another axiom's effect
                        format_pne(&ref_ax.effect.name, &ref_ax.effect.args)
                    }
                };
                let idx = *num_index.get(&part_key).ok_or_else(|| {
                    format!(
                        "numeric axiom part not found: {} for axiom {}",
                        part_key, ax.effect.name
                    )
                })?;
                part_indices.push(idx);
            }

            // Get the operator
            let op = match &ax.op {
                Some(op_str) => op_str.clone(),
                None => {
                    let is_constant = ax.parts.len() == 1
                        && matches!(ax.parts[0], crate::translate::numeric_axiom_rules::NumericPart::Constant(_));
                    if is_constant {
                        continue;
                    }
                    return Err(format!("numeric axiom has no op: {}", ax.effect.name));
                }
            };

            numeric_axioms.push(crate::translate::sas::NumericAxiom {
                op,
                parts: part_indices,
                effect: effect_idx,
            });
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

    // Goal is represented by a dedicated goal axiom variable
    let goal_pairs: Vec<(usize, usize)> = vec![(goal_axiom_var, 0)];

    // Process axiom conditions to ensure comparison axiom variables are created
    // This must happen BEFORE we try to look up comparison variables in collect_condition_pairs
    {
        fn ensure_axiom_comparison_vars(
            cond: &crate::translate::pddl_ast::Condition,
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
            comp_axiom_added: &mut std::collections::HashSet<usize>,
            normalize_op: &dyn Fn(&str) -> String,
            negate_op: &dyn Fn(&str) -> String,
        ) {
            match cond {
                crate::translate::pddl_ast::Condition::Comparison(opstr, l, r) => {
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
                            let op_norm = normalize_op(&opstr);
                            let comp_key = format!("{} {} {}", op_norm, ni, nj);
                            let effect_idx = if let Some(&ei) = var_index.get(&comp_key) {
                                ei
                            } else {
                                let ei = vars.len();
                                var_index.insert(comp_key.clone(), ei);
                                let pos = format!("{} {} {}", op_norm, ni, nj);
                                let neg = format!("{} {} {}", negate_op(&op_norm), ni, nj);
                                vars.push(crate::translate::sas::Variable {
                                    value_names: vec![pos, neg, "<none of those>".to_string()],
                                });
                                ranges.push(3);
                                ei
                            };
                            // Only add the comparison axiom if we haven't seen this effect_var yet
                            if !comp_axiom_added.contains(&effect_idx) {
                                comp_axiom_added.insert(effect_idx);
                                comp_axioms.push(crate::translate::sas::CompareAxiom {
                                    comp: normalize_op(&opstr),
                                    parts: vec![ni, nj],
                                    effect_var: effect_idx,
                                });
                            }
                        }
                    }
                }
                crate::translate::pddl_ast::Condition::And(list) => {
                    for c in list {
                        ensure_axiom_comparison_vars(
                            c,
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
                            comp_axiom_added,
                            normalize_op,
                            negate_op,
                        );
                    }
                }
                _ => {}
            }
        }
        
        // Process all propositional axiom conditions
        for ax in propositional_axioms {
            ensure_axiom_comparison_vars(
                &ax.condition,
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
                &mut comp_axiom_added,
                &normalize_op,
                &negate_op,
            );
        }

        // Process goal condition comparisons for goal axiom
        ensure_axiom_comparison_vars(
            normalized_goal,
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
            &mut comp_axiom_added,
            &normalize_op,
            &negate_op,
        );
    }

    let comp_var_set: std::collections::HashSet<usize> =
        comp_axioms.iter().map(|c| c.effect_var).collect();
    let axiom_var_set: std::collections::HashSet<usize> =
        axiom_atom_to_var.values().copied().collect();

    let canonical_variables: Vec<crate::translate::sas::CanonicalVariable> = vars
        .iter()
        .enumerate()
        .map(|(idx, v)| {
            let axiom_layer = if axiom_var_set.contains(&idx) {
                30
            } else if comp_var_set.contains(&idx) {
                29
            } else {
                -1
            };
            crate::translate::sas::CanonicalVariable {
                name: format!("var{}", idx),
                axiom_layer,
                values: v.value_names.clone(),
            }
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

    // Build SASAxioms from propositional_axioms
    // Each axiom has a condition that needs to be converted to (var, val) pairs
    let mut sas_axioms: Vec<crate::translate::sas::SASAxiom> = Vec::new();
    fn collect_condition_pairs(
        cond: &crate::translate::pddl_ast::Condition,
        atom_to_fdr: &std::collections::HashMap<String, (usize, usize)>,
        axiom_atom_to_var: &std::collections::HashMap<String, usize>,
        var_index: &std::collections::HashMap<String, usize>,
        num_index: &std::collections::HashMap<String, usize>,
        ranges: &[usize],
        condition_pairs: &mut Vec<(usize, usize)>,
        normalize_op: &dyn Fn(&str) -> String,
    ) {
                match cond {
                    crate::translate::pddl_ast::Condition::Atom(name, args) => {
                        let atom = format!("{}({})", name, args.join(", "));
                        if let Some(&v) = axiom_atom_to_var.get(&atom) {
                            condition_pairs.push((v, 0)); // 0 = true
                        } else if let Some(&(v, val)) = atom_to_fdr.get(&atom) {
                            condition_pairs.push((v, val));
                        }
                    }
                    crate::translate::pddl_ast::Condition::Comparison(opstr, l, r) => {
                        // The comparison should have been converted to a comparison axiom variable
                        // Look it up by its string representation
                        fn sexpr_to_name(s: &crate::translate::pddl_parser::SExpr) -> Option<String> {
                            match s {
                                crate::translate::pddl_parser::SExpr::Atom(a) => {
                                    if a.starts_with("derived!") {
                                        Some(format!("{}()", a))
                                    } else {
                                        Some(format!("{}()", a))
                                    }
                                }
                                crate::translate::pddl_parser::SExpr::List(list) => {
                                    if list.is_empty() { return None; }
                                    if let crate::translate::pddl_parser::SExpr::Atom(name) = &list[0] {
                                        let args: Vec<String> = list[1..].iter()
                                            .filter_map(|s| if let crate::translate::pddl_parser::SExpr::Atom(a) = s { Some(a.clone()) } else { None })
                                            .collect();
                                        Some(format!("{}({})", name, args.join(", ")))
                                    } else { None }
                                }
                            }
                        }
                        if let (Some(left_key), Some(right_key)) = (sexpr_to_name(l), sexpr_to_name(r)) {
                            if let (Some(&ni), Some(&nj)) = (num_index.get(&left_key), num_index.get(&right_key)) {
                                let op_norm = normalize_op(opstr);
                                let comp_key = format!("{} {} {}", op_norm, ni, nj);
                                if let Some(&comp_var) = var_index.get(&comp_key) {
                                    condition_pairs.push((comp_var, 0)); // 0 = comparison true
                                }
                            }
                        }
                    }
                    crate::translate::pddl_ast::Condition::And(parts) => {
                        for p in parts {
                            collect_condition_pairs(
                                p,
                                atom_to_fdr,
                                axiom_atom_to_var,
                                var_index,
                                num_index,
                                ranges,
                                condition_pairs,
                                normalize_op,
                            );
                        }
                    }
                    crate::translate::pddl_ast::Condition::Not(inner) => {
                        // For negated atoms, value 1 = false
                        if let crate::translate::pddl_ast::Condition::Atom(name, args) = &**inner {
                            let atom = format!("{}({})", name, args.join(", "));
                            if let Some(&v) = axiom_atom_to_var.get(&atom) {
                                condition_pairs.push((v, 1)); // 1 = false/negated
                            } else if let Some(&(v, _)) = atom_to_fdr.get(&atom) {
                                // For regular atoms, we need to find the negated value
                                // Typically the last value is "none of those"
                                condition_pairs.push((v, ranges[v] - 1));
                            }
                        }
                    }
                    _ => {}
                }
    }

    for ax in propositional_axioms {
        let ax_name = axiom_name_map.get(&ax.name).cloned().unwrap_or_else(|| ax.name.clone());
        let atom = format!("{}()", ax_name);
        if let Some(&effect_var) = axiom_atom_to_var.get(&atom) {
            let mut condition_pairs: Vec<(usize, usize)> = Vec::new();
            collect_condition_pairs(
                &ax.condition,
                &atom_to_fdr,
                &axiom_atom_to_var,
                &var_index,
                &num_index,
                &ranges,
                &mut condition_pairs,
                &normalize_op,
            );
            sas_axioms.push(crate::translate::sas::SASAxiom {
                condition: condition_pairs,
                effect: (effect_var, 0), // effect_val 0 = true
            });
        }
    }

    // Add goal axiom using the normalized goal condition
    {
        let mut goal_condition_pairs: Vec<(usize, usize)> = Vec::new();
        collect_condition_pairs(
            normalized_goal,
            &atom_to_fdr,
            &axiom_atom_to_var,
            &var_index,
            &num_index,
            &ranges,
            &mut goal_condition_pairs,
            &normalize_op,
        );
        sas_axioms.push(crate::translate::sas::SASAxiom {
            condition: goal_condition_pairs,
            effect: (goal_axiom_var, 0),
        });
    }

    // We don't yet populate comparison axioms from other sources here; leave empty for now.
    Ok(SASTask {
        variables: vars,
        operators,
        numeric_variables: numeric_list,
        numeric_axioms,
        comparison_axioms: comp_axioms,
        axioms: sas_axioms,
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
    })
}
