use std::collections::{HashMap, HashSet};
use std::path::Path;

use crate::translate::build_model::Atom as ModelAtom;
use crate::translate::instantiate;
use crate::translate::normalize;
use crate::translate::numeric_axiom_rules::{InstantiatedNumericAxiom, PrimitiveNumericExpression};
use crate::translate::pddl;
use crate::translate::pddl::{Atom, FunctionComparison, Literal, NegatedFunctionComparison};
use crate::translate::pddl_parser::PddlTask;
use crate::translate::simplify;
use crate::translate::options;
use crate::translate::sas as internal_sas;
use crate::translate::sas_tasks as py_sas_tasks;
use internal_sas::{CompareAxiom, SASAxiom, SASOperator, SASTask as InternalSASTask};
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Debug, Clone)]
pub struct TranslateConfig {
    pub simplify: bool,
}

static SIMPLIFIED_EFFECT_CONDITION_COUNTER: AtomicUsize = AtomicUsize::new(0);
static ADDED_IMPLIED_PRECONDITION_COUNTER: AtomicUsize = AtomicUsize::new(0);

impl Default for TranslateConfig {
    fn default() -> Self {
        Self { simplify: true }
    }
}

pub fn translate_from_files(domain: &Path, problem: &Path) -> Result<py_sas_tasks::SASTask, String> {
    translate_from_files_with_config(domain, problem, &TranslateConfig::default())
}

pub fn translate_from_files_with_config(
    domain: &Path,
    problem: &Path,
    config: &TranslateConfig,
) -> Result<py_sas_tasks::SASTask, String> {
    let task = PddlTask::from_files(domain, problem).map_err(|err| err.to_string())?;
    let dom = pddl::Domain::from_sexprs(&task.domain_forms)
        .ok_or_else(|| "failed to parse domain PDDL".to_string())?;
    let prob = pddl::Problem::from_sexprs(&task.problem_forms)
        .ok_or_else(|| "failed to parse problem PDDL".to_string())?;
    translate_from_ast(&dom, &prob, config)
}

pub fn translate_from_ast(
    dom: &pddl::Domain,
    prob: &pddl::Problem,
    config: &TranslateConfig,
) -> Result<py_sas_tasks::SASTask, String> {
    let mut norm_task = normalize::NormalizableTask::from_ast(dom, prob);
    norm_task.add_global_constraints();
    normalize::normalize(&mut norm_task)?;

    let exploration = instantiate::explore_normalized(&norm_task)
        .map_err(|err| err.to_string())?;

    let py_groups: Option<Vec<Vec<String>>> = None;
    let mut sas_task = translate_task_from_grounded_internal(
        &exploration.grounded_ops,
        dom,
        prob,
        &exploration.numeric_axioms,
        py_groups,
        &exploration.grounded_axioms,
        &norm_task.goal,
        &norm_task,
    )
    .map_err(|err| err.to_string())?;

    if config.simplify {
        match simplify::filter_unreachable_propositions(&mut sas_task) {
            Ok(_) => {}
            Err(simplify::SimplifyError::Impossible) => {
                sas_task = simplify::trivial_task(false);
            }
            Err(simplify::SimplifyError::TriviallySolvable) => {
                sas_task = simplify::trivial_task(true);
            }
        }
    }

    Ok(py_sas_tasks::from_internal(&sas_task))
}

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
    let format_pne_key = |name: &str, args: &[String]| -> String {
        if args.is_empty() {
            format!("{}()", name)
        } else {
            format!("{}({})", name, args.join(", "))
        }
    };
    match sexpr {
        crate::translate::pddl_parser::SExpr::List(inner) => {
            if inner.is_empty() {
                return None;
            }
            if let crate::translate::pddl_parser::SExpr::Atom(op) = &inner[0] {
                if op == "+" || op == "-" || op == "*" || op == "/" {
                    let pne = df_admin.get_derived_function(
                        &crate::translate::pddl_parser::SExpr::List(inner.clone()),
                    );
                    let derived_key = format_pne_key(&pne.name, &pne.args);
                    let mut parts_numericparts: Vec<
                        crate::translate::numeric_axiom_rules::NumericPart,
                    > = Vec::new();
                    for p in &inner[1..] {
                        match p {
                            crate::translate::pddl_parser::SExpr::Atom(a) => {
                                if let Ok(nv) = a.parse::<i64>() {
                                    parts_numericparts.push(
                                        crate::translate::numeric_axiom_rules::NumericPart::Constant(
                                            crate::translate::numeric_axiom_rules::NumericConstant(
                                                nv,
                                            ),
                                        ),
                                    );
                                } else {
                                    let prim = crate::translate::numeric_axiom_rules::PrimitiveNumericExpression {
                                        name: a.clone(),
                                        args: vec![],
                                    };
                                    parts_numericparts.push(
                                        crate::translate::numeric_axiom_rules::NumericPart::Primitive(
                                            prim,
                                        ),
                                    );
                                }
                            }
                            crate::translate::pddl_parser::SExpr::List(_) => {
                                let child = df_admin.get_derived_function(p);
                                let prim = crate::translate::numeric_axiom_rules::PrimitiveNumericExpression {
                                    name: child.name.clone(),
                                    args: child.args.clone(),
                                };
                                parts_numericparts.push(
                                    crate::translate::numeric_axiom_rules::NumericPart::Primitive(
                                        prim,
                                    ),
                                );
                            }
                        }
                    }
                    if !num_index.contains_key(&derived_key) {
                        let idx = numeric_list.len();
                        num_index.insert(derived_key.clone(), idx);
                        numeric_list.push(crate::translate::sas::NumericVariable {
                            name: derived_key.clone(),
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
                            name: derived_key.clone(),
                            op: Some(op.clone()),
                            parts: parts_numericparts,
                            effect,
                        };
                        let ai = instantiated_num_axioms.len();
                        instantiated_num_axioms.push(ax.clone());
                        derived_axiom_index.insert(derived_key.clone(), ai);
                    }
                    return Some(derived_key);
                }
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
                let const_symbol = format!("derived!{}.0()", v);
                let const_key = format_pne_key(&const_symbol, &[]);
                if !num_index.contains_key(&const_key) {
                    let idx = numeric_list.len();
                    num_index.insert(const_key.clone(), idx);
                    numeric_list.push(crate::translate::sas::NumericVariable {
                        name: const_key.clone(),
                        initial: Some(v),
                        ntype: "C".to_string(),
                        axiom_layer: -1,
                    });
                    numeric_init_vec.push(v);
                    let pne = crate::translate::numeric_axiom_rules::PrimitiveNumericExpression {
                        name: const_symbol.clone(),
                        args: vec![],
                    };
                    let part = crate::translate::numeric_axiom_rules::NumericPart::Constant(
                        crate::translate::numeric_axiom_rules::NumericConstant(v),
                    );
                    let ax = crate::translate::numeric_axiom_rules::InstantiatedNumericAxiom {
                        name: const_key.clone(),
                        op: None,
                        parts: vec![part],
                        effect: pne,
                    };
                    let ai = instantiated_num_axioms.len();
                    instantiated_num_axioms.push(ax);
                    derived_axiom_index.insert(const_key.clone(), ai);
                }
                Some(const_key)
            } else {
                let key = format!("{}()", a);
                if num_index.contains_key(&key) {
                    Some(key)
                } else if num_index.contains_key(a) {
                    Some(a.clone())
                } else {
                    None
                }
            }
        }
    }
}

pub fn translate_task_from_grounded_internal(
    ops: &[crate::translate::instantiate::GroundedOp],
    dom: &crate::translate::pddl::Domain,
    prob: &crate::translate::pddl::Problem,
    external_instantiated_num_axioms: &Vec<
        crate::translate::numeric_axiom_rules::InstantiatedNumericAxiom,
    >,
    py_groups: Option<Vec<Vec<String>>>,
    grounded_axioms: &[crate::translate::instantiate::GroundedAxiom],
    normalized_goal: &crate::translate::pddl::Condition,
    norm_task: &crate::translate::normalize::NormalizableTask,
) -> Result<InternalSASTask, String> {
    fn normalize_op(op: &str) -> String {
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
    let mut atom_to_fdr: std::collections::HashMap<String, (usize, usize)> =
        std::collections::HashMap::new();
    let mut vars: Vec<crate::translate::sas::Variable> = Vec::new();
    let mut ranges: Vec<usize> = Vec::new();
    let mut fact_to_varvals: HashMap<String, Vec<(usize, usize)>> = HashMap::new();
    let mut instantiated_num_axioms: Vec<
        crate::translate::numeric_axiom_rules::InstantiatedNumericAxiom,
    > = external_instantiated_num_axioms.clone();
    let mut derived_axiom_index: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();

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
    let mut fluent_preds: std::collections::HashSet<String> = std::collections::HashSet::new();
    for act in &dom.actions {
        if let Some(eff_s) = &act.effect {
            let eff = crate::translate::pddl::sexpr_to_effect(eff_s);
            fn collect(
                e: &crate::translate::pddl::Effect,
                set: &mut std::collections::HashSet<String>,
            ) {
                match e {
                    crate::translate::pddl::Effect::Add(n, _)
                    | crate::translate::pddl::Effect::Del(n, _) => {
                        set.insert(n.clone());
                    }
                    crate::translate::pddl::Effect::And(v) => {
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

    let mut numeric_inits: Vec<(String, i64)> = Vec::new();
    let mut grounded_atoms: Vec<String> = Vec::new();
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
                                        if func_names.contains(fname) {
                                            numeric_inits.push((key, n));
                                        } else {
                                            grounded_atoms.push(key.clone());
                                            numeric_inits.push((key, n));
                                        }
                                    }
                                }
                            }
                        }
                    } else if let crate::translate::pddl_parser::SExpr::Atom(name) = &list[0] {
                        if fluent_preds.contains(name.as_str()) {
                            let atom = format!(
                                "{}({})",
                                name,
                                list[1..]
                                    .iter()
                                    .filter_map(|x| match x {
                                        crate::translate::pddl_parser::SExpr::Atom(s) => {
                                            Some(s.clone())
                                        }
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

    let mut numeric_vars: Vec<(String, i64)> = Vec::new();
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
                    crate::translate::pddl::Effect::Add(name, args) => {
                        grounded_atoms.push(format!("{}({})", name, args.join(", ")));
                    }
                    crate::translate::pddl::Effect::Del(name, args) => {
                        grounded_atoms.push(format!("{}({})", name, args.join(", ")));
                    }
                    crate::translate::pddl::Effect::Increase(name, args, val) => {
                        let key = format!("{}({})", name, args.join(", "));
                        numeric_vars.push((key, *val));
                    }
                    crate::translate::pddl::Effect::Decrease(name, args, val) => {
                        let key = format!("{}({})", name, args.join(", "));
                        numeric_vars.push((key, -*val));
                    }
                    crate::translate::pddl::Effect::And(_) => {}
                }
            }
        } else if let Some(eff) = &op.eff {
            match eff {
                crate::translate::pddl::Effect::Add(name, args) => {
                    grounded_atoms.push(format!("{}({})", name, args.join(", ")));
                }
                crate::translate::pddl::Effect::Del(name, args) => {
                    grounded_atoms.push(format!("{}({})", name, args.join(", ")));
                }
                crate::translate::pddl::Effect::And(v) => {
                    for sub in v {
                        match sub {
                            crate::translate::pddl::Effect::Add(name, args) => {
                                grounded_atoms.push(format!("{}({})", name, args.join(", ")));
                            }
                            crate::translate::pddl::Effect::Del(name, args) => {
                                grounded_atoms.push(format!("{}({})", name, args.join(", ")));
                            }
                            crate::translate::pddl::Effect::Increase(name, args, val) => {
                                let key = format!("{}({})", name, args.join(", "));
                                numeric_vars.push((key, *val));
                            }
                            crate::translate::pddl::Effect::Decrease(name, args, val) => {
                                let key = format!("{}({})", name, args.join(", "));
                                numeric_vars.push((key, -*val));
                            }
                            crate::translate::pddl::Effect::And(_) => {}
                        }
                    }
                }
                crate::translate::pddl::Effect::Increase(name, args, v) => {
                    let key = format!("{}({})", name, args.join(", "));
                    numeric_vars.push((key, *v));
                }
                crate::translate::pddl::Effect::Decrease(name, args, v) => {
                    let key = format!("{}({})", name, args.join(", "));
                    numeric_vars.push((key, -*v));
                }
            }
        }
        if let Some(pre) = &op.pre {
            match pre {
                crate::translate::pddl::Condition::Atom(name, args) => {
                    if fluent_preds.contains(name.as_str()) {
                        grounded_atoms.push(format!("{}({})", name, args.join(", ")));
                    }
                }
                crate::translate::pddl::Condition::And(v) => {
                    for c in v {
                        match c {
                            crate::translate::pddl::Condition::Atom(name, args) => {
                                if fluent_preds.contains(name.as_str()) {
                                    grounded_atoms
                                        .push(format!("{}({})", name, args.join(", ")));
                                }
                            }
                            crate::translate::pddl::Condition::Comparison(_, _, _) => {}
                            crate::translate::pddl::Condition::Not(_) => {}
                            crate::translate::pddl::Condition::And(_) => {}
                            crate::translate::pddl::Condition::Or(_) => {}
                            crate::translate::pddl::Condition::Forall(_, _) => {}
                            crate::translate::pddl::Condition::Exists(_, _) => {}
                            crate::translate::pddl::Condition::True => {}
                        }
                    }
                }
                crate::translate::pddl::Condition::Comparison(_, _, _) => {}
                crate::translate::pddl::Condition::Not(_) => {}
                crate::translate::pddl::Condition::Or(_) => {}
                crate::translate::pddl::Condition::Forall(_, _) => {}
                crate::translate::pddl::Condition::Exists(_, _) => {}
                crate::translate::pddl::Condition::True => {}
            }
        }
    }

    fn collect_atoms_from_condition(
        cond: &crate::translate::pddl::Condition,
        grounded_atoms: &mut Vec<String>,
        fluent_preds: &std::collections::HashSet<String>,
    ) {
        match cond {
            crate::translate::pddl::Condition::Atom(name, args) => {
                if fluent_preds.contains(name.as_str()) {
                    grounded_atoms.push(format!("{}({})", name, args.join(", ")));
                }
            }
            crate::translate::pddl::Condition::Not(inner) => {
                collect_atoms_from_condition(inner, grounded_atoms, fluent_preds);
            }
            crate::translate::pddl::Condition::And(list) => {
                for c in list {
                    collect_atoms_from_condition(c, grounded_atoms, fluent_preds);
                }
            }
            _ => {}
        }
    }

    for ax in grounded_axioms {
        collect_atoms_from_condition(&ax.condition, &mut grounded_atoms, &fluent_preds);
        grounded_atoms.push(ax.effect_atom.clone());
    }

    let (chosen_groups, _mutex_groups, translation_key) = if let Some(pg) = py_groups {
        (Vec::new(), Vec::new(), pg)
    } else {
        crate::translate::fact_groups::compute_groups(norm_task, &grounded_atoms, None)
    };
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

    let mut mutex_groups_pairs: Vec<Vec<(usize, usize)>> = Vec::new();
    if !chosen_groups.is_empty() {
        for group in chosen_groups {
            let mg: Vec<(usize, usize)> = group
                .into_iter()
                .filter_map(|atom| atom_to_fdr.get(&atom).cloned())
                .collect();
            if mg.len() > 1 {
                mutex_groups_pairs.push(mg);
            }
        }
    }

    let mut numeric_list: Vec<crate::translate::sas::NumericVariable> = Vec::new();
    let mut numeric_init_vec: Vec<i64> = Vec::new();
    let mut num_index: std::collections::HashMap<String, usize> = std::collections::HashMap::new();

    let constant_axioms =
        crate::translate::numeric_axiom_rules::identify_constants_inplace(
            &mut instantiated_num_axioms,
        );
    let axiom_by_pne =
        crate::translate::numeric_axiom_rules::axiom_by_pne(&instantiated_num_axioms);
    let constant_effects: std::collections::HashSet<_> =
        constant_axioms.iter().map(|a| &a.effect).collect();
    let (axioms_by_layer, _max_layer) = crate::translate::numeric_axiom_rules::compute_axiom_layers(
        &instantiated_num_axioms,
        &constant_axioms,
        &axiom_by_pne,
    );

    let axiom_map = crate::translate::numeric_axiom_rules::identify_equivalent_axioms(
        &axioms_by_layer,
        &axiom_by_pne,
    );

    let mut axiom_effects_added: std::collections::HashSet<String> =
        std::collections::HashSet::new();
    let mut redundant_axioms: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();

    let mut sorted_axioms: Vec<_> = instantiated_num_axioms.iter().collect();
    sorted_axioms.sort_by(|a, b| {
        let a_name = format_pne(&a.effect);
        let b_name = format_pne(&b.effect);
        a_name.cmp(&b_name)
    });

    for axiom in sorted_axioms.iter() {
        let effect_name = format_pne(&axiom.effect);
        if let Some(mapped_axiom) = axiom_map.get(&axiom.effect) {
            let mapped_name = format_pne(&mapped_axiom.effect);
            redundant_axioms.insert(effect_name.clone(), mapped_name);
            continue;
        }

        if !axiom_effects_added.contains(&effect_name) {
            axiom_effects_added.insert(effect_name.clone());

            let ntype = if constant_effects.contains(&axiom.effect) {
                "C".to_string()
            } else {
                "D".to_string()
            };

            let axiom_layer: i32 = -1;

            let idx = numeric_list.len();
            num_index.insert(effect_name.clone(), idx);
            numeric_list.push(crate::translate::sas::NumericVariable {
                name: effect_name,
                initial: Some(0),
                ntype,
                axiom_layer,
            });
            numeric_init_vec.push(0);
        }
    }

    for (redundant_name, target_name) in &redundant_axioms {
        if let Some(&target_idx) = num_index.get(target_name) {
            num_index.insert(redundant_name.clone(), target_idx);
        }
    }

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
    num_map.entry("total-cost()".to_string()).or_insert(0);

    let mut fluent_entries: Vec<(String, i64)> =
        num_map.iter().map(|(k, v)| (k.clone(), *v)).collect();
    fluent_entries.sort_by(|a, b| a.0.cmp(&b.0));
    for (n, v) in fluent_entries.iter() {
        if num_index.contains_key(n) {
            continue;
        }

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

    let mut operators: Vec<crate::translate::sas::SASOperator> = Vec::new();
    let mut numeric_axioms: Vec<crate::translate::sas::NumericAxiom> = Vec::new();
    let mut comp_axioms: Vec<crate::translate::sas::CompareAxiom> = Vec::new();
    let mut comp_axiom_added: std::collections::HashSet<usize> = std::collections::HashSet::new();
    let _comp_index: std::collections::HashMap<(String, Vec<usize>, usize), usize> =
        std::collections::HashMap::new();
    for op in ops {
        let mut prevails: Vec<(usize, usize)> = Vec::new();
        let mut effects: Vec<(usize, usize, usize, Vec<(usize, usize)>)> = Vec::new();
        let mut op_numeric_effects: Vec<(usize, String, usize, Vec<(usize, usize)>)> = Vec::new();

        if let Some(pre) = &op.pre {
            match pre {
                crate::translate::pddl::Condition::Atom(name, args) => {
                    let atom = format!("{}({})", name, args.join(", "));
                    if let Some(&(v, val)) = atom_to_fdr.get(&atom) {
                        prevails.push((v, val));
                    }
                }
                crate::translate::pddl::Condition::And(v) => {
                    for c in v {
                        match c {
                            crate::translate::pddl::Condition::Atom(name, args) => {
                                let atom = format!("{}({})", name, args.join(", "));
                                if let Some(&(v, val)) = atom_to_fdr.get(&atom) {
                                    prevails.push((v, val));
                                }
                            }
                            crate::translate::pddl::Condition::Comparison(opstr, l, r) => {
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
                                        let op_norm = normalize_op(&opstr);
                                        let comp_key = format!("{} {} {}", op_norm, ni, nj);
                                        let effect_idx = if let Some(&ei) = var_index.get(&comp_key)
                                        {
                                            ei
                                        } else {
                                            let ei = vars.len();
                                            var_index.insert(comp_key.clone(), ei);
                                            let pos = format!("{} {} {}", op_norm, ni, nj);
                                            let neg =
                                                format!("{} {} {}", negate_op(&op_norm), ni, nj);
                                            vars.push(crate::translate::sas::Variable {
                                                value_names: vec![
                                                    pos,
                                                    neg,
                                                    "<none of those>".to_string(),
                                                ],
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
                                        prevails.push((effect_idx, 0));
                                    }
                                }
                            }
                            crate::translate::pddl::Condition::Not(_) => {}
                            crate::translate::pddl::Condition::And(_) => {}
                            crate::translate::pddl::Condition::Or(_) => {}
                            crate::translate::pddl::Condition::Forall(_, _) => {}
                            crate::translate::pddl::Condition::Exists(_, _) => {}
                            crate::translate::pddl::Condition::True => {}
                        }
                    }
                }
                crate::translate::pddl::Condition::Comparison(opstr, l, r) => {
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
                    let mut build_condition_vals =
                        |conds: &[crate::translate::pddl::Condition]| {
                            let mut cond_vals: Vec<(usize, usize)> = Vec::new();
                            let mut stack: Vec<crate::translate::pddl::Condition> =
                                conds.to_vec();
                            while let Some(cond) = stack.pop() {
                                match cond {
                                    crate::translate::pddl::Condition::Atom(name, args) => {
                                        let atom = format!("{}({})", name, args.join(", "));
                                        if let Some(&(v, val)) = atom_to_fdr.get(&atom) {
                                            cond_vals.push((v, val));
                                        }
                                    }
                                    crate::translate::pddl::Condition::Not(inner) => {
                                        if let crate::translate::pddl::Condition::Atom(
                                            name,
                                            args,
                                        ) = *inner
                                        {
                                            let atom = format!("{}({})", name, args.join(", "));
                                            let neg = format!("NegatedAtom {}", atom);
                                            if let Some(&(v, val)) = atom_to_fdr.get(&neg) {
                                                cond_vals.push((v, val));
                                            }
                                        }
                                    }
                                    crate::translate::pddl::Condition::And(parts) => {
                                        for part in parts {
                                            stack.push(part);
                                        }
                                    }
                                    crate::translate::pddl::Condition::Comparison(opstr, l, r) => {
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
                                        if let (Some(left_key), Some(right_key)) =
                                            (left_name, right_name)
                                        {
                                            if let (Some(&ni), Some(&nj)) = (
                                                num_index.get(&left_key),
                                                num_index.get(&right_key),
                                            ) {
                                                let comp_key =
                                                    format!("{} {} {}", opstr, left_key, right_key);
                                                let effect_idx = if let Some(&ei) =
                                                    var_index.get(&comp_key)
                                                {
                                                    ei
                                                } else {
                                                    let ei = vars.len();
                                                    var_index.insert(comp_key.clone(), ei);
                                                    let pos = format!(
                                                        "{} {} {}",
                                                        opstr, left_key, right_key
                                                    );
                                                    let neg = format!(
                                                        "not {} {} {}",
                                                        opstr, left_key, right_key
                                                    );
                                                    vars.push(crate::translate::sas::Variable {
                                                        value_names: vec![
                                                            pos,
                                                            neg,
                                                            "<none of those>".to_string(),
                                                        ],
                                                    });
                                                    ranges.push(3);
                                                    ei
                                                };
                                                if !comp_axiom_added.contains(&effect_idx) {
                                                    comp_axiom_added.insert(effect_idx);
                                                    comp_axioms.push(
                                                        crate::translate::sas::CompareAxiom {
                                                            comp: normalize_op(&opstr),
                                                            parts: vec![ni, nj],
                                                            effect_var: effect_idx,
                                                        },
                                                    );
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
                    crate::translate::pddl::Effect::Add(name, args) => {
                        let atom = format!("{}({})", name, args.join(", "));
                        if let Some(&(v, val)) = atom_to_fdr.get(&atom) {
                            let pre = ranges.get(v).map(|r| r - 1).unwrap_or(0);
                            effects.push((v, pre, val, condition.clone()));
                        }
                    }
                    crate::translate::pddl::Effect::Del(name, args) => {
                        let atom = format!("{}({})", name, args.join(", "));
                        if let Some(&(v, val)) = atom_to_fdr.get(&atom) {
                            let none_idx = ranges[v] - 1;
                            effects.push((v, val, none_idx, condition.clone()));
                        }
                    }
                    crate::translate::pddl::Effect::Increase(nname, args, val) => {
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
                            op_numeric_effects.push((
                                ni,
                                "+".to_string(),
                                rhs_idx,
                                condition.clone(),
                            ));
                        }
                    }
                    crate::translate::pddl::Effect::Decrease(nname, args, val) => {
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
                            op_numeric_effects.push((
                                ni,
                                "-".to_string(),
                                rhs_idx,
                                condition.clone(),
                            ));
                        }
                    }
                    crate::translate::pddl::Effect::And(_) => {}
                }
            }
        } else if let Some(eff) = &op.eff {
            match eff {
                crate::translate::pddl::Effect::Add(name, args) => {
                    let atom = format!("{}({})", name, args.join(", "));
                    if let Some(&(v, val)) = atom_to_fdr.get(&atom) {
                        let pre = ranges.get(v).map(|r| r - 1).unwrap_or(0);
                        effects.push((v, pre, val, vec![]));
                    }
                }
                crate::translate::pddl::Effect::Del(name, args) => {
                    let atom = format!("{}({})", name, args.join(", "));
                    if let Some(&(v, val)) = atom_to_fdr.get(&atom) {
                        let none_idx = ranges[v] - 1;
                        effects.push((v, val, none_idx, vec![]));
                    }
                }
                crate::translate::pddl::Effect::And(v) => {
                    for sub in v {
                        match sub {
                            crate::translate::pddl::Effect::Add(name, args) => {
                                let atom = format!("{}({})", name, args.join(", "));
                                if let Some(&(v, val)) = atom_to_fdr.get(&atom) {
                                    let pre = ranges.get(v).map(|r| r - 1).unwrap_or(0);
                                    effects.push((v, pre, val, vec![]));
                                }
                            }
                            crate::translate::pddl::Effect::Del(name, args) => {
                                let atom = format!("{}({})", name, args.join(", "));
                                if let Some(&(v, val)) = atom_to_fdr.get(&atom) {
                                    let none_idx = ranges[v] - 1;
                                    effects.push((v, val, none_idx, vec![]));
                                }
                            }
                            crate::translate::pddl::Effect::Increase(nname, args, val) => {
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
                                    op_numeric_effects
                                        .push((ni, "+".to_string(), rhs_idx, vec![]));
                                }
                            }
                            crate::translate::pddl::Effect::Decrease(nname, args, val) => {
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
                                    op_numeric_effects
                                        .push((ni, "-".to_string(), rhs_idx, vec![]));
                                }
                            }
                            crate::translate::pddl::Effect::And(_) => {}
                        }
                    }
                }
                crate::translate::pddl::Effect::Increase(nname, args, val) => {
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
                crate::translate::pddl::Effect::Decrease(nname, args, val) => {
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

    let (num_axioms_by_layer, _max_layer, num_axiom_map, _const_num_axioms) =
        crate::translate::numeric_axiom_rules::handle_axioms_checked(&instantiated_num_axioms)
            .map_err(|err| format!("numeric axiom error: {}", err))?;

    {
        let pne_key = |name: &str, args: &[String]| -> String {
            if args.is_empty() {
                format!("{}()", name)
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
                                if ok {
                                    acc
                                } else {
                                    continue;
                                }
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

    {
        let pne_key = |name: &str, args: &[String]| -> String {
            if args.is_empty() {
                format!("{}()", name)
            } else {
                format!("{}({})", name, args.join(", "))
            }
        };
        let mut layers: Vec<i32> = num_axioms_by_layer.keys().copied().collect();
        layers.sort();
        let mut num_axiom_layer: i32 = 0;
        for layer in layers {
            if let Some(axs) = num_axioms_by_layer.get(&layer) {
                let mut sorted_axs = axs.clone();
                sorted_axs.sort_by(|a, b| a.name.cmp(&b.name));
                for ax in sorted_axs {
                    if num_axiom_map.contains_key(&ax.effect) {
                        continue;
                    }
                    let effect_key = pne_key(&ax.effect.name, &ax.effect.args);
                    if let Some(&idx) = num_index.get(&effect_key) {
                        if layer == -1 {
                            numeric_list[idx].axiom_layer = -1;
                        } else {
                            numeric_list[idx].axiom_layer = num_axiom_layer;
                            num_axiom_layer += 1;
                        }
                    }
                }
            }
        }
    }

    let format_pne = |name: &str, args: &[String]| -> String {
        if args.is_empty() {
            format!("{}()", name)
        } else {
            format!("{}({})", name, args.join(", "))
        }
    };
    for (_layer, axs) in &num_axioms_by_layer {
        for ax in axs {
            let effect_key = format_pne(&ax.effect.name, &ax.effect.args);
            let effect_idx = *num_index
                .get(&effect_key)
                .ok_or_else(|| format!("numeric axiom effect not found: {}", effect_key))?;

            let is_constant = ax.op.is_none()
                && ax.parts.len() == 1
                && matches!(
                    ax.parts[0],
                    crate::translate::numeric_axiom_rules::NumericPart::Constant(_)
                );
            if is_constant {
                continue;
            }

            let mut part_indices: Vec<usize> = Vec::new();
            for part in &ax.parts {
                let part_key = match part {
                    crate::translate::numeric_axiom_rules::NumericPart::Primitive(pne) => {
                        format_pne(&pne.name, &pne.args)
                    }
                    crate::translate::numeric_axiom_rules::NumericPart::Constant(c) => {
                        format!("derived!{}.0()", c.0)
                    }
                    crate::translate::numeric_axiom_rules::NumericPart::Axiom(ref_ax) => {
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

            let op = match &ax.op {
                Some(op_str) => op_str.clone(),
                None => return Err(format!("numeric axiom has no op: {}", ax.effect.name)),
            };

            numeric_axioms.push(crate::translate::sas::NumericAxiom {
                op,
                parts: part_indices,
                effect: effect_idx,
            });
        }
    }

    {
        fn ensure_axiom_comparison_vars(
            cond: &crate::translate::pddl::Condition,
            ensure_expr_var: &mut dyn FnMut(
                &crate::translate::pddl_parser::SExpr,
                &mut std::collections::HashMap<String, usize>,
                &mut Vec<crate::translate::sas::NumericVariable>,
                &mut Vec<i64>,
                &mut Vec<crate::translate::numeric_axiom_rules::InstantiatedNumericAxiom>,
                &mut std::collections::HashMap<String, usize>,
            ) -> Option<String>,
            num_index: &mut std::collections::HashMap<String, usize>,
            numeric_list: &mut Vec<crate::translate::sas::NumericVariable>,
            numeric_init_vec: &mut Vec<i64>,
            instantiated_num_axioms: &mut Vec<
                crate::translate::numeric_axiom_rules::InstantiatedNumericAxiom,
            >,
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
                crate::translate::pddl::Condition::Comparison(opstr, l, r) => {
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
                        if let (Some(&ni), Some(&nj)) =
                            (num_index.get(&left_key), num_index.get(&right_key))
                        {
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
                crate::translate::pddl::Condition::And(list) => {
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

        for ax in grounded_axioms {
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

    let comp_var_set: std::collections::HashSet<usize> =
        comp_axioms.iter().map(|c| c.effect_var).collect();

    let canonical_variables: Vec<crate::translate::sas::CanonicalVariable> = vars
        .iter()
        .enumerate()
        .map(|(idx, v)| {
            let axiom_layer = if comp_var_set.contains(&idx) { 29 } else { -1 };
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
                .map(
                    |(var, pre, post, cond)| crate::translate::sas::CanonicalEffect {
                        var: *var,
                        pre: Some(*pre),
                        post: *post,
                        condition: cond.clone(),
                    },
                )
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

    let mut sas_axioms: Vec<crate::translate::sas::SASAxiom> = Vec::new();
    fn collect_condition_pairs(
        cond: &crate::translate::pddl::Condition,
        atom_to_fdr: &std::collections::HashMap<String, (usize, usize)>,
        var_index: &std::collections::HashMap<String, usize>,
        num_index: &std::collections::HashMap<String, usize>,
        ranges: &[usize],
        condition_pairs: &mut Vec<(usize, usize)>,
        normalize_op: &dyn Fn(&str) -> String,
    ) {
        match cond {
            crate::translate::pddl::Condition::Atom(name, args) => {
                let atom = format!("{}({})", name, args.join(", "));
                if let Some(&(v, val)) = atom_to_fdr.get(&atom) {
                    condition_pairs.push((v, val));
                }
            }
            crate::translate::pddl::Condition::Comparison(opstr, l, r) => {
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
                            if list.is_empty() {
                                return None;
                            }
                            if let crate::translate::pddl_parser::SExpr::Atom(name) = &list[0] {
                                let args: Vec<String> = list[1..]
                                    .iter()
                                    .filter_map(|s| {
                                        if let crate::translate::pddl_parser::SExpr::Atom(a) = s {
                                            Some(a.clone())
                                        } else {
                                            None
                                        }
                                    })
                                    .collect();
                                Some(format!("{}({})", name, args.join(", ")))
                            } else {
                                None
                            }
                        }
                    }
                }
                if let (Some(left_key), Some(right_key)) = (sexpr_to_name(l), sexpr_to_name(r)) {
                    if let (Some(&ni), Some(&nj)) =
                        (num_index.get(&left_key), num_index.get(&right_key))
                    {
                        let op_norm = normalize_op(opstr);
                        let comp_key = format!("{} {} {}", op_norm, ni, nj);
                        if let Some(&comp_var) = var_index.get(&comp_key) {
                            condition_pairs.push((comp_var, 0));
                        }
                    }
                }
            }
            crate::translate::pddl::Condition::And(parts) => {
                for p in parts {
                    collect_condition_pairs(
                        p,
                        atom_to_fdr,
                        var_index,
                        num_index,
                        ranges,
                        condition_pairs,
                        normalize_op,
                    );
                }
            }
            crate::translate::pddl::Condition::Not(inner) => {
                if let crate::translate::pddl::Condition::Atom(name, args) = &**inner {
                    let atom = format!("{}({})", name, args.join(", "));
                    if let Some(&(v, _)) = atom_to_fdr.get(&atom) {
                        condition_pairs.push((v, ranges[v] - 1));
                    }
                }
            }
            _ => {}
        }
    }

    let mut goal_pairs: Vec<(usize, usize)> = Vec::new();
    collect_condition_pairs(
        normalized_goal,
        &atom_to_fdr,
        &var_index,
        &num_index,
        &ranges,
        &mut goal_pairs,
        &normalize_op,
    );

    for ax in grounded_axioms {
        if let Some(&(effect_var, effect_val)) = atom_to_fdr.get(&ax.effect_atom) {
            let mut condition_pairs: Vec<(usize, usize)> = Vec::new();
            collect_condition_pairs(
                &ax.condition,
                &atom_to_fdr,
                &var_index,
                &num_index,
                &ranges,
                &mut condition_pairs,
                &normalize_op,
            );
            sas_axioms.push(crate::translate::sas::SASAxiom {
                condition: condition_pairs,
                effect: (effect_var, effect_val),
            });
        }
    }

    let global_constraint = match norm_task.global_constraint.as_ref() {
        None => None,
        Some(name) => {
            let atom = format!("{}()", name);
            let (var, val) = atom_to_fdr
                .get(&atom)
                .copied()
                .ok_or_else(|| format!("global constraint atom not found: {}", atom))?;
            Some((var, val))
        }
    };
    Ok(InternalSASTask {
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
        global_constraint,
        comp_axiom_layer: 0,
    })
}

pub fn translate_task_from_grounded(
    ops: &[crate::translate::instantiate::GroundedOp],
    dom: &crate::translate::pddl::Domain,
    prob: &crate::translate::pddl::Problem,
    external_instantiated_num_axioms: &Vec<
        crate::translate::numeric_axiom_rules::InstantiatedNumericAxiom,
    >,
    py_groups: Option<Vec<Vec<String>>>,
    grounded_axioms: &[crate::translate::instantiate::GroundedAxiom],
    normalized_goal: &crate::translate::pddl::Condition,
    norm_task: &crate::translate::normalize::NormalizableTask,
) -> Result<py_sas_tasks::SASTask, String> {
    let task = translate_task_from_grounded_internal(
        ops,
        dom,
        prob,
        external_instantiated_num_axioms,
        py_groups,
        grounded_axioms,
        normalized_goal,
        norm_task,
    )?;
    Ok(py_sas_tasks::from_internal(&task))
}

fn format_pne(pne: &PrimitiveNumericExpression) -> String {
    if pne.args.is_empty() {
        format!("{}()", pne.name)
    } else {
        format!("{}({})", pne.name, pne.args.join(", "))
    }
}

pub fn strips_to_sas_dictionary(
    groups: &[Vec<ModelAtom>],
    num_axioms: &[InstantiatedNumericAxiom],
    num_axiom_map: &HashMap<PrimitiveNumericExpression, InstantiatedNumericAxiom>,
    num_fluents: &HashSet<String>,
    assert_partial: bool,
    include_numeric: bool,
) -> (
    Vec<usize>,
    HashMap<ModelAtom, Vec<(usize, usize)>>,
    usize,
    HashMap<String, usize>,
) {
    let mut dictionary: HashMap<ModelAtom, Vec<(usize, usize)>> = HashMap::new();
    for (var_no, group) in groups.iter().enumerate() {
        for (val_no, atom) in group.iter().enumerate() {
            dictionary
                .entry(atom.clone())
                .or_default()
                .push((var_no, val_no));
        }
    }
    if assert_partial {
        for pairs in dictionary.values() {
            assert!(pairs.len() == 1);
        }
    }

    let ranges: Vec<usize> = groups.iter().map(|g| g.len() + 1).collect();

    let mut numeric_dictionary: HashMap<String, usize> = HashMap::new();
    let mut num_count = 0;

    if include_numeric {
        let mut redundant_axioms: Vec<PrimitiveNumericExpression> = Vec::new();
        for axiom in num_axioms {
            if num_axiom_map.contains_key(&axiom.effect) {
                redundant_axioms.push(axiom.effect.clone());
            } else {
                let key = format_pne(&axiom.effect);
                numeric_dictionary.insert(key, num_count);
                num_count += 1;
            }
        }

        for axiom_effect in redundant_axioms {
            if let Some(mapped) = num_axiom_map.get(&axiom_effect) {
                let key = format_pne(&axiom_effect);
                let mapped_key = format_pne(&mapped.effect);
                if let Some(idx) = numeric_dictionary.get(&mapped_key) {
                    numeric_dictionary.insert(key, *idx);
                }
            }
        }

        let mut fluent_list: Vec<String> = num_fluents.iter().cloned().collect();
        fluent_list.sort();
        for fluent in fluent_list {
            if !numeric_dictionary.contains_key(&fluent) {
                numeric_dictionary.insert(fluent, num_count);
                num_count += 1;
            }
        }
    }

    (ranges, dictionary, num_count, numeric_dictionary)
}

pub fn build_mutex_key(
    strips_to_sas: &HashMap<ModelAtom, Vec<(usize, usize)>>,
    groups: &[Vec<ModelAtom>],
) -> Vec<Vec<(usize, usize)>> {
    let mut group_keys = Vec::new();
    for group in groups {
        let mut group_key = Vec::new();
        for fact in group {
            if let Some(entries) = strips_to_sas.get(fact) {
                for (var, val) in entries {
                    group_key.push((*var, *val));
                }
            }
        }
        group_keys.push(group_key);
    }
    group_keys
}

pub fn build_implied_facts(
    strips_to_sas: &HashMap<ModelAtom, Vec<(usize, usize)>>,
    groups: &[Vec<ModelAtom>],
    mutex_groups: &[Vec<ModelAtom>],
) -> HashMap<(usize, usize), Vec<(usize, usize)>> {
    let mut lonely_propositions: HashMap<ModelAtom, usize> = HashMap::new();
    for (var_no, group) in groups.iter().enumerate() {
        if group.len() == 1 {
            let lonely_prop = group[0].clone();
            lonely_propositions.insert(lonely_prop, var_no);
        }
    }

    let mut implied: HashMap<(usize, usize), Vec<(usize, usize)>> = HashMap::new();
    for mutex_group in mutex_groups {
        for prop in mutex_group {
            if let Some(prop_var) = lonely_propositions.get(prop) {
                let prop_is_false = (*prop_var, 1);
                for other_prop in mutex_group {
                    if other_prop != prop {
                        if let Some(other_facts) = strips_to_sas.get(other_prop) {
                            for other_fact in other_facts {
                                implied
                                    .entry(*other_fact)
                                    .or_default()
                                    .push(prop_is_false);
                            }
                        }
                    }
                }
            }
        }
    }

    implied
}

pub fn dump_statistics(task: &py_sas_tasks::SASTask) {
    let derived_vars = task
        .variables
        .axiom_layers
        .iter()
        .filter(|layer| **layer >= 0)
        .count();
    let total_facts: usize = task.variables.ranges.iter().sum();
    let mutex_groups = task.mutexes.len();
    let mutex_size: usize = task.mutexes.iter().map(|g| g.facts.len()).sum();

    println!("Translator variables: {}", task.variables.ranges.len());
    println!("Translator derived variables: {}", derived_vars);
    println!("Translator facts: {}", total_facts);
    println!("Translator goal facts: {}", task.goal.pairs.len());
    println!("Translator mutex groups: {}", mutex_groups);
    println!("Translator total mutex groups size: {}", mutex_size);
    println!("Translator operators: {}", task.operators.len());
    println!("Translator axioms: {}", task.axioms.len());
    println!("Translator task size: {}", task.variables.ranges.len());
}

pub fn translate_strips_conditions_aux(
    conditions: &[Literal],
    dictionary: &mut HashMap<Literal, Vec<(usize, usize)>>,
    ranges: &mut Vec<usize>,
    numeric_dictionary: &HashMap<String, usize>,
    comp_axiom_dict: &mut HashMap<(String, Vec<usize>), Literal>,
    sas_comp_axioms: &mut Vec<CompareAxiom>,
    mutex_check: bool,
) -> Option<Vec<HashMap<usize, usize>>> {
    let mut condition: HashMap<usize, HashSet<usize>> = HashMap::new();

    for fact in conditions {
        match fact {
            Literal::FunctionComparison(comp) => {
                let negated = false;
                let comp = comp.clone();
                if !dictionary.contains_key(fact) {
                    for part in &comp.parts {
                        assert!(numeric_dictionary.contains_key(part));
                    }
                    let parts: Vec<usize> = comp
                        .parts
                        .iter()
                        .map(|p| numeric_dictionary[p])
                        .collect();
                    let key = (comp.comparator.clone(), parts.clone());
                    let mut fact_to_use = Literal::FunctionComparison(comp.clone());
                    if let Some(pos_fact) = comp_axiom_dict.get(&key) {
                        fact_to_use = if negated {
                            pos_fact.negate()
                        } else {
                            pos_fact.clone()
                        };
                    } else {
                        let axiom = CompareAxiom {
                            comp: comp.comparator.clone(),
                            parts: parts.clone(),
                            effect_var: ranges.len(),
                        };
                        let pos_fact = Literal::FunctionComparison(comp.clone());
                        let neg_fact = Literal::NegatedFunctionComparison(NegatedFunctionComparison {
                            comparator: comp.comparator.clone(),
                            parts: comp.parts.clone(),
                        });
                        if !mutex_check {
                            sas_comp_axioms.push(axiom);
                            comp_axiom_dict.insert(key, pos_fact.clone());
                        }
                        dictionary
                            .entry(pos_fact)
                            .or_default()
                            .push((ranges.len(), 0));
                        dictionary
                            .entry(neg_fact)
                            .or_default()
                            .push((ranges.len(), 1));
                        ranges.push(3);
                    }

                    if let Some(entry) = dictionary.get(&fact_to_use).and_then(|v| v.first()) {
                        let (var, val) = *entry;
                        if let Some(existing) = condition.get(&var) {
                            if !existing.contains(&val) {
                                return None;
                            }
                        }
                        condition.insert(var, HashSet::from([val]));
                    }
                } else if let Some(entry) = dictionary.get(fact).and_then(|v| v.first()) {
                    let (var, val) = *entry;
                    if let Some(existing) = condition.get(&var) {
                        if !existing.contains(&val) {
                            return None;
                        }
                    }
                    condition.insert(var, HashSet::from([val]));
                }
            }
            Literal::NegatedFunctionComparison(comp) => {
                let negated = true;
                let comp = FunctionComparison {
                    comparator: comp.comparator.clone(),
                    parts: comp.parts.clone(),
                };
                if !dictionary.contains_key(fact) {
                    for part in &comp.parts {
                        assert!(numeric_dictionary.contains_key(part));
                    }
                    let parts: Vec<usize> = comp
                        .parts
                        .iter()
                        .map(|p| numeric_dictionary[p])
                        .collect();
                    let key = (comp.comparator.clone(), parts.clone());
                    let mut fact_to_use = Literal::NegatedFunctionComparison(NegatedFunctionComparison {
                        comparator: comp.comparator.clone(),
                        parts: comp.parts.clone(),
                    });
                    if let Some(pos_fact) = comp_axiom_dict.get(&key) {
                        fact_to_use = if negated {
                            pos_fact.negate()
                        } else {
                            pos_fact.clone()
                        };
                    } else {
                        let axiom = CompareAxiom {
                            comp: comp.comparator.clone(),
                            parts: parts.clone(),
                            effect_var: ranges.len(),
                        };
                        let pos_fact = Literal::FunctionComparison(comp.clone());
                        let neg_fact = Literal::NegatedFunctionComparison(NegatedFunctionComparison {
                            comparator: comp.comparator.clone(),
                            parts: comp.parts.clone(),
                        });
                        if !mutex_check {
                            sas_comp_axioms.push(axiom);
                            comp_axiom_dict.insert(key, pos_fact.clone());
                        }
                        dictionary
                            .entry(pos_fact)
                            .or_default()
                            .push((ranges.len(), 0));
                        dictionary
                            .entry(neg_fact)
                            .or_default()
                            .push((ranges.len(), 1));
                        ranges.push(3);
                    }

                    if let Some(entry) = dictionary.get(&fact_to_use).and_then(|v| v.first()) {
                        let (var, val) = *entry;
                        if let Some(existing) = condition.get(&var) {
                            if !existing.contains(&val) {
                                return None;
                            }
                        }
                        condition.insert(var, HashSet::from([val]));
                    }
                } else if let Some(entry) = dictionary.get(fact).and_then(|v| v.first()) {
                    let (var, val) = *entry;
                    if let Some(existing) = condition.get(&var) {
                        if !existing.contains(&val) {
                            return None;
                        }
                    }
                    condition.insert(var, HashSet::from([val]));
                }
            }
            Literal::Atom(_) => {
                if let Some(entries) = dictionary.get(fact) {
                    for (var, val) in entries {
                        if let Some(existing) = condition.get(var) {
                            if !existing.contains(val) {
                                return None;
                            }
                        }
                        condition.insert(*var, HashSet::from([*val]));
                    }
                }
            }
            Literal::NegatedAtom(_) => {
                continue;
            }
        }
    }

    let number_of_values = |vals: &HashSet<usize>| vals.len();

    for fact in conditions {
        if matches!(
            fact,
            Literal::FunctionComparison(_) | Literal::NegatedFunctionComparison(_)
        ) {
            continue;
        }
        if fact.is_negated() {
            let mut done = false;
            let mut new_condition: HashMap<usize, HashSet<usize>> = HashMap::new();
            if let Some(atom) = fact.positive_atom() {
                let positive_fact = Literal::Atom(atom);
                if let Some(entries) = dictionary.get(&positive_fact) {
                    for (var, val) in entries {
                        let mut poss_vals: HashSet<usize> =
                            (0..ranges[*var]).collect();
                        poss_vals.remove(val);

                        if !condition.contains_key(var) {
                            if new_condition.contains_key(var) {
                                continue;
                            }
                            new_condition.insert(*var, poss_vals);
                        } else if let Some(existing) = condition.get_mut(var) {
                            done = true;
                            existing.retain(|v| poss_vals.contains(v));
                            if existing.is_empty() {
                                return None;
                            }
                        }
                    }
                }
            }

            if !done && !new_condition.is_empty() {
                let mut candidates: Vec<(usize, HashSet<usize>)> = new_condition
                    .into_iter()
                    .collect();
                candidates.sort_by_key(|(_, vals)| number_of_values(vals));
                if let Some((var, vals)) = candidates.into_iter().next() {
                    condition.insert(var, vals);
                }
            }
        }
    }

    let mut sorted_conds: Vec<(usize, HashSet<usize>)> = condition.into_iter().collect();
    sorted_conds.sort_by_key(|(_, vals)| number_of_values(vals));

    let mut flat_conds: Vec<HashMap<usize, usize>> = vec![HashMap::new()];
    for (var, vals) in sorted_conds {
        if vals.len() == 1 {
            let val = *vals.iter().next().unwrap();
            for cond in &mut flat_conds {
                cond.insert(var, val);
            }
        } else {
            let mut new_conds = Vec::new();
            for cond in &flat_conds {
                for val in &vals {
                    let mut new_cond = cond.clone();
                    new_cond.insert(var, *val);
                    new_conds.push(new_cond);
                }
            }
            flat_conds = new_conds;
        }
    }

    Some(flat_conds)
}

pub fn translate_strips_conditions(
    conditions: &[Literal],
    dictionary: &mut HashMap<Literal, Vec<(usize, usize)>>,
    ranges: &mut Vec<usize>,
    numeric_dictionary: &HashMap<String, usize>,
    mutex_dict: &mut HashMap<Literal, Vec<(usize, usize)>>,
    mutex_ranges: &mut Vec<usize>,
    comp_axiom_dict: &mut HashMap<(String, Vec<usize>), Literal>,
    sas_comp_axioms: &mut Vec<CompareAxiom>,
) -> Option<Vec<HashMap<usize, usize>>> {
    if conditions.is_empty() {
        return Some(vec![HashMap::new()]);
    }

    if translate_strips_conditions_aux(
        conditions,
        mutex_dict,
        mutex_ranges,
        numeric_dictionary,
        comp_axiom_dict,
        sas_comp_axioms,
        true,
    )
    .is_none()
    {
        return None;
    }

    translate_strips_conditions_aux(
        conditions,
        dictionary,
        ranges,
        numeric_dictionary,
        comp_axiom_dict,
        sas_comp_axioms,
        false,
    )
}

#[derive(Clone, Debug)]
pub struct AssignmentEffect {
    pub fluent: String,
    pub symbol: String,
    pub expression: String,
}

#[derive(Clone, Debug)]
pub struct StripsOperator {
    pub name: String,
    pub precondition: Vec<Literal>,
    pub add_effects: Vec<(Vec<Literal>, Literal)>,
    pub del_effects: Vec<(Vec<Literal>, Literal)>,
    pub assign_effects: Vec<(Vec<Literal>, AssignmentEffect)>,
    pub cost: f64,
}

#[derive(Clone, Debug)]
pub struct StripsAxiom {
    pub condition: Vec<Literal>,
    pub effect: Literal,
}

pub fn negate_and_translate_condition(
    condition: &[Vec<Literal>],
    dictionary: &mut HashMap<Literal, Vec<(usize, usize)>>,
    ranges: &mut Vec<usize>,
    numeric_dict: &HashMap<String, usize>,
    mutex_dict: &mut HashMap<Literal, Vec<(usize, usize)>>,
    mutex_ranges: &mut Vec<usize>,
    comp_axiom_dict: &mut HashMap<(String, Vec<usize>), Literal>,
    sas_comp_axioms: &mut Vec<CompareAxiom>,
) -> Option<Vec<HashMap<usize, usize>>> {
    if condition.iter().any(|clause| clause.is_empty()) {
        return None;
    }

    let mut combinations: Vec<Vec<Literal>> = vec![Vec::new()];
    for clause in condition {
        let mut next = Vec::new();
        for combo in &combinations {
            for lit in clause {
                let mut new_combo = combo.clone();
                new_combo.push(lit.clone());
                next.push(new_combo);
            }
        }
        combinations = next;
    }

    let mut negation = Vec::new();
    for combination in combinations {
        let negated: Vec<Literal> = combination
            .iter()
            .map(|lit| lit.negate())
            .collect();
        if let Some(cond) = translate_strips_conditions(
            &negated,
            dictionary,
            ranges,
            numeric_dict,
            mutex_dict,
            mutex_ranges,
            comp_axiom_dict,
            sas_comp_axioms,
        ) {
            negation.extend(cond);
        }
    }
    if negation.is_empty() {
        None
    } else {
        Some(negation)
    }
}

pub fn translate_strips_operator(
    operator: &StripsOperator,
    dictionary: &mut HashMap<Literal, Vec<(usize, usize)>>,
    ranges: &mut Vec<usize>,
    numeric_dictionary: &HashMap<String, usize>,
    mutex_dict: &mut HashMap<Literal, Vec<(usize, usize)>>,
    mutex_ranges: &mut Vec<usize>,
    implied_facts: &HashMap<(usize, usize), Vec<(usize, usize)>>,
    comp_axiom_dict: &mut HashMap<(String, Vec<usize>), Literal>,
    sas_comp_axioms: &mut Vec<CompareAxiom>,
    num_vals: usize,
    relevant_numeric_vars: &HashSet<usize>,
) -> Vec<SASOperator> {
    let conditions = translate_strips_conditions(
        &operator.precondition,
        dictionary,
        ranges,
        numeric_dictionary,
        mutex_dict,
        mutex_ranges,
        comp_axiom_dict,
        sas_comp_axioms,
    );
    let mut sas_operators = Vec::new();
    if let Some(condition_list) = conditions {
        for condition in condition_list {
            if let Some(op) = translate_strips_operator_aux(
                operator,
                dictionary,
                ranges,
                numeric_dictionary,
                mutex_dict,
                mutex_ranges,
                implied_facts,
                condition,
                comp_axiom_dict,
                sas_comp_axioms,
                num_vals,
                relevant_numeric_vars,
            ) {
                sas_operators.push(op);
            }
        }
    }
    sas_operators
}

pub fn translate_strips_operator_aux(
    operator: &StripsOperator,
    dictionary: &mut HashMap<Literal, Vec<(usize, usize)>>,
    ranges: &mut Vec<usize>,
    numeric_dictionary: &HashMap<String, usize>,
    mutex_dict: &mut HashMap<Literal, Vec<(usize, usize)>>,
    mutex_ranges: &mut Vec<usize>,
    implied_facts: &HashMap<(usize, usize), Vec<(usize, usize)>>,
    mut condition: HashMap<usize, usize>,
    comp_axiom_dict: &mut HashMap<(String, Vec<usize>), Literal>,
    sas_comp_axioms: &mut Vec<CompareAxiom>,
    _num_vals: usize,
    relevant_numeric: &HashSet<usize>,
) -> Option<SASOperator> {
    let mut effects_by_variable: HashMap<usize, HashMap<usize, Vec<HashMap<usize, usize>>>> =
        HashMap::new();
    let mut add_conds_by_variable: HashMap<usize, Vec<Vec<Literal>>> = HashMap::new();

    for (conds, fact) in &operator.add_effects {
        let eff_condition_list = translate_strips_conditions(
            conds,
            dictionary,
            ranges,
            numeric_dictionary,
            mutex_dict,
            mutex_ranges,
            comp_axiom_dict,
            sas_comp_axioms,
        )?;
        if let Some(entries) = dictionary.get(fact) {
            for (var, val) in entries {
                effects_by_variable
                    .entry(*var)
                    .or_default()
                    .entry(*val)
                    .or_default()
                    .extend(eff_condition_list.clone());
                add_conds_by_variable
                    .entry(*var)
                    .or_default()
                    .push(conds.clone());
            }
        }
    }

    let mut del_effects_by_variable: HashMap<usize, HashMap<usize, Vec<HashMap<usize, usize>>>> =
        HashMap::new();
    for (conds, fact) in &operator.del_effects {
        let eff_condition_list = translate_strips_conditions(
            conds,
            dictionary,
            ranges,
            numeric_dictionary,
            mutex_dict,
            mutex_ranges,
            comp_axiom_dict,
            sas_comp_axioms,
        )?;
        if let Some(entries) = dictionary.get(fact) {
            for (var, val) in entries {
                del_effects_by_variable
                    .entry(*var)
                    .or_default()
                    .entry(*val)
                    .or_default()
                    .extend(eff_condition_list.clone());
            }
        }
    }

    let mut ass_effects_by_variable: HashMap<usize, HashMap<(String, usize), Vec<HashMap<usize, usize>>>> =
        HashMap::new();
    for (conds, assignment) in &operator.assign_effects {
        let eff_condition_list = translate_strips_conditions(
            conds,
            dictionary,
            ranges,
            numeric_dictionary,
            mutex_dict,
            mutex_ranges,
            comp_axiom_dict,
            sas_comp_axioms,
        )?;
        if let Some(expr_idx) = numeric_dictionary.get(&assignment.expression) {
            if let Some(fluent_idx) = numeric_dictionary.get(&assignment.fluent) {
                ass_effects_by_variable
                    .entry(*fluent_idx)
                    .or_default()
                    .entry((assignment.symbol.clone(), *expr_idx))
                    .or_default()
                    .extend(eff_condition_list);
            }
        }
    }

    for (var, del_effects) in del_effects_by_variable {
        let no_add_effect_condition = negate_and_translate_condition(
            add_conds_by_variable.get(&var).map(|v| v.as_slice()).unwrap_or(&[]),
            dictionary,
            ranges,
            numeric_dictionary,
            mutex_dict,
            mutex_ranges,
            comp_axiom_dict,
            sas_comp_axioms,
        );
        if no_add_effect_condition.is_none() {
            continue;
        }
        let no_add_effect_condition = no_add_effect_condition.unwrap();
        let none_of_those = ranges[var] - 1;
        for (val, conds) in del_effects {
            for mut cond in conds {
                if let Some(existing) = cond.get(&var) {
                    if *existing != val {
                        continue;
                    }
                }
                cond.insert(var, val);
                for no_add_cond in &no_add_effect_condition {
                    let mut new_cond = cond.clone();
                    let mut ok = true;
                    for (cvar, cval) in no_add_cond {
                        if let Some(existing) = new_cond.get(cvar) {
                            if *existing != *cval {
                                ok = false;
                                break;
                            }
                        }
                        new_cond.insert(*cvar, *cval);
                    }
                    if ok {
                        effects_by_variable
                            .entry(var)
                            .or_default()
                            .entry(none_of_those)
                            .or_default()
                            .push(new_cond);
                    }
                }
            }
        }
    }

    build_sas_operator(
        &operator.name,
        &mut condition,
        effects_by_variable,
        ass_effects_by_variable,
        operator.cost,
        ranges,
        implied_facts,
        relevant_numeric,
    )
}

pub fn build_sas_operator(
    name: &str,
    condition: &mut HashMap<usize, usize>,
    effects_by_variable: HashMap<usize, HashMap<usize, Vec<HashMap<usize, usize>>>>,
    ass_effects_by_variable: HashMap<usize, HashMap<(String, usize), Vec<HashMap<usize, usize>>>>,
    deprecated_cost: f64,
    ranges: &[usize],
    implied_facts: &HashMap<(usize, usize), Vec<(usize, usize)>>,
    relevant_numeric_variables: &HashSet<usize>,
) -> Option<SASOperator> {
    let add_implied = options::get()
        .map(|opts| opts.add_implied_preconditions)
        .unwrap_or(false);
    let mut implied_precondition: HashSet<(usize, usize)> = HashSet::new();
    if add_implied {
        for (var, val) in condition.iter() {
            if let Some(implied) = implied_facts.get(&(*var, *val)) {
                implied_precondition.extend(implied);
            }
        }
    }

    let mut prevail_and_pre = condition.clone();
    let mut pre_post = Vec::new();
    let mut num_pre_post = Vec::new();

    for (var, effects) in effects_by_variable {
        let orig_pre = *condition.get(&var).unwrap_or(&usize::MAX);
        let mut added_effect = false;
        for (post, eff_conditions) in effects {
            let mut pre = orig_pre;
            if pre == post {
                continue;
            }
            let mut eff_condition_lists: Vec<Vec<(usize, usize)>> = eff_conditions
                .iter()
                .map(|eff_cond| {
                    let mut v: Vec<(usize, usize)> = eff_cond.iter().map(|(k, v)| (*k, *v)).collect();
                    v.sort();
                    v
                })
                .collect();
            if ranges[var] == 2 {
                if prune_stupid_effect_conditions(var, post, &mut eff_condition_lists) {
                    SIMPLIFIED_EFFECT_CONDITION_COUNTER.fetch_add(1, Ordering::Relaxed);
                }
                if add_implied && pre == usize::MAX && implied_precondition.contains(&(var, 1 - post)) {
                    ADDED_IMPLIED_PRECONDITION_COUNTER.fetch_add(1, Ordering::Relaxed);
                    pre = 1 - post;
                }
            }
            for eff_condition in eff_condition_lists {
                let mut filtered = Vec::new();
                let mut contradicts = false;
                for (variable, value) in eff_condition {
                    if let Some(prev) = prevail_and_pre.get(&variable) {
                        if *prev != value {
                            contradicts = true;
                            break;
                        }
                    } else {
                        filtered.push((variable, value));
                    }
                }
                if contradicts {
                    continue;
                }
                pre_post.push((var, if pre == usize::MAX { 0 } else { pre }, post, filtered));
                added_effect = true;
            }
        }
        if added_effect {
            condition.remove(&var);
        }
    }

    for (numvar, effects) in ass_effects_by_variable {
        for ((ass_op, post_var), eff_conditions) in effects {
            let eff_condition_lists: Vec<Vec<(usize, usize)>> = eff_conditions
                .iter()
                .map(|eff_cond| {
                    let mut v: Vec<(usize, usize)> = eff_cond.iter().map(|(k, v)| (*k, *v)).collect();
                    v.sort();
                    v
                })
                .collect();
            for eff_condition in eff_condition_lists {
                num_pre_post.push((numvar, ass_op.clone(), post_var, eff_condition));
            }
        }
    }

    if pre_post.is_empty() {
        let mut irrelevant = true;
        for effect in &num_pre_post {
            if relevant_numeric_variables.contains(&effect.0) {
                irrelevant = false;
                break;
            }
        }
        if irrelevant {
            return None;
        }
    }

    let prevail = condition.iter().map(|(k, v)| (*k, *v)).collect();
    Some(SASOperator {
        name: name.to_string(),
        prevails: prevail,
        effects: pre_post,
        numeric_effects: num_pre_post,
        cost: deprecated_cost,
    })
}

pub fn prune_stupid_effect_conditions(
    var: usize,
    val: usize,
    conditions: &mut Vec<Vec<(usize, usize)>>,
) -> bool {
    if conditions == &vec![Vec::new()] {
        return false;
    }
    let dual_fact = (var, 1 - val);
    let mut simplified = false;
    for condition in conditions.iter_mut() {
        while let Some(pos) = condition.iter().position(|v| *v == dual_fact) {
            condition.remove(pos);
            simplified = true;
        }
        if condition.is_empty() {
            conditions.clear();
            conditions.push(Vec::new());
            simplified = true;
            break;
        }
    }
    simplified
}

pub fn translate_strips_axiom(
    axiom: &StripsAxiom,
    dictionary: &mut HashMap<Literal, Vec<(usize, usize)>>,
    ranges: &mut Vec<usize>,
    num_dict: &HashMap<String, usize>,
    mutex_dict: &mut HashMap<Literal, Vec<(usize, usize)>>,
    mutex_ranges: &mut Vec<usize>,
    comp_axiom_dict: &mut HashMap<(String, Vec<usize>), Literal>,
    sas_comp_axioms: &mut Vec<CompareAxiom>,
) -> Vec<SASAxiom> {
    let conditions = translate_strips_conditions(
        &axiom.condition,
        dictionary,
        ranges,
        num_dict,
        mutex_dict,
        mutex_ranges,
        comp_axiom_dict,
        sas_comp_axioms,
    );
    let mut axioms = Vec::new();
    if let Some(cond_list) = conditions {
        let effect = if axiom.effect.is_negated() {
            if let Some(atom) = axiom.effect.positive_atom() {
                let positive = Literal::Atom(atom);
                if let Some(entries) = dictionary.get(&positive) {
                    let (var, _) = entries[0];
                    (var, ranges[var] - 1)
                } else {
                    return axioms;
                }
            } else {
                return axioms;
            }
        } else if let Some(entries) = dictionary.get(&axiom.effect) {
            entries[0]
        } else {
            return axioms;
        };
        for condition in cond_list {
            axioms.push(SASAxiom {
                condition: condition.iter().map(|(k, v)| (*k, *v)).collect(),
                effect,
            });
        }
    }
    axioms
}

pub fn translate_numeric_axiom(
    axiom: &InstantiatedNumericAxiom,
    prop_dictionary: &HashMap<Literal, Vec<(usize, usize)>>,
    num_dictionary: &HashMap<String, usize>,
) -> Option<crate::translate::sas::NumericAxiom> {
    let effect = num_dictionary.get(&format_pne(&axiom.effect)).copied()?;
    let mut parts = Vec::new();
    for part in &axiom.parts {
        match part {
            crate::translate::numeric_axiom_rules::NumericPart::Primitive(pne) => {
                parts.push(*num_dictionary.get(&format_pne(pne))?);
            }
            crate::translate::numeric_axiom_rules::NumericPart::Axiom(sub) => {
                let key = format_pne(&sub.effect);
                if let Some(idx) = num_dictionary.get(&key) {
                    parts.push(*idx);
                } else if let Some(entries) = prop_dictionary.get(&Literal::Atom(Atom {
                    predicate: key,
                    args: Vec::new(),
                })) {
                    parts.push(entries[0].0);
                } else {
                    return None;
                }
            }
            crate::translate::numeric_axiom_rules::NumericPart::Constant(constant) => {
                let key = constant.0.to_string();
                parts.push(*num_dictionary.get(&key)?);
            }
        }
    }

    Some(crate::translate::sas::NumericAxiom {
        op: axiom.op.clone().unwrap_or_default(),
        parts,
        effect,
    })
}

pub fn translate_strips_operators(
    actions: &[StripsOperator],
    strips_to_sas: &mut HashMap<Literal, Vec<(usize, usize)>>,
    ranges: &mut Vec<usize>,
    numeric_strips_to_sas: &HashMap<String, usize>,
    mutex_dict: &mut HashMap<Literal, Vec<(usize, usize)>>,
    mutex_ranges: &mut Vec<usize>,
    implied_facts: &HashMap<(usize, usize), Vec<(usize, usize)>>,
    comp_axiom_dict: &mut HashMap<(String, Vec<usize>), Literal>,
    sas_comp_axioms: &mut Vec<CompareAxiom>,
    num_vals: usize,
    relevant_numeric_vars: &HashSet<usize>,
) -> Vec<SASOperator> {
    let mut result = Vec::new();
    for action in actions {
        let mut ops = translate_strips_operator(
            action,
            strips_to_sas,
            ranges,
            numeric_strips_to_sas,
            mutex_dict,
            mutex_ranges,
            implied_facts,
            comp_axiom_dict,
            sas_comp_axioms,
            num_vals,
            relevant_numeric_vars,
        );
        result.append(&mut ops);
    }
    result
}

pub fn translate_strips_axioms(
    axioms: &[StripsAxiom],
    strips_to_sas: &mut HashMap<Literal, Vec<(usize, usize)>>,
    ranges: &mut Vec<usize>,
    num_dict: &HashMap<String, usize>,
    mutex_dict: &mut HashMap<Literal, Vec<(usize, usize)>>,
    mutex_ranges: &mut Vec<usize>,
    comp_axiom_dict: &mut HashMap<(String, Vec<usize>), Literal>,
    sas_comp_axioms: &mut Vec<CompareAxiom>,
) -> Vec<SASAxiom> {
    let mut result = Vec::new();
    for axiom in axioms {
        let mut sas_axioms = translate_strips_axiom(
            axiom,
            strips_to_sas,
            ranges,
            num_dict,
            mutex_dict,
            mutex_ranges,
            comp_axiom_dict,
            sas_comp_axioms,
        );
        result.append(&mut sas_axioms);
    }
    result
}

pub fn add_key_to_comp_axioms(
    comparison_axioms: &mut Vec<CompareAxiom>,
    translation_key: &mut Vec<Vec<String>>,
) {
    for axiom in comparison_axioms {
        let mut value_list = Vec::new();
        value_list.push(format!("{} {:?}", axiom.comp, axiom.parts));
        value_list.push(format!("not({} {:?})", axiom.comp, axiom.parts));
        value_list.push("<none of those>".to_string());
        translation_key.push(value_list);
    }
}

pub fn dump_task(
    init: &[Literal],
    goals: &[Literal],
    actions: &[StripsOperator],
    axioms: &[StripsAxiom],
    axiom_layer_dict: &HashMap<String, usize>,
) -> Result<(), String> {
    let mut out = String::new();
    out.push_str("Initial state\n");
    for atom in init {
        out.push_str(&format!("{:?}\n", atom));
    }
    out.push_str("\nGoals\n");
    for goal in goals {
        out.push_str(&format!("{:?}\n", goal));
    }
    for action in actions {
        out.push_str("\nAction\n");
        out.push_str(&format!("{:?}\n", action));
    }
    for axiom in axioms {
        out.push_str("\nAxiom\n");
        out.push_str(&format!("{:?}\n", axiom));
    }
    out.push_str("\nAxiom layers\n");
    for (atom, layer) in axiom_layer_dict {
        out.push_str(&format!("{}: layer {}\n", atom, layer));
    }
    std::fs::write("output.dump", out)
        .map_err(|err| format!("failed to write output.dump: {}", err))
}

fn solvable_sas_task_internal() -> InternalSASTask {
    simplify::trivial_task(true)
}

fn unsolvable_sas_task_internal() -> InternalSASTask {
    simplify::trivial_task(false)
}

pub fn trivial_task(solvable: bool) -> py_sas_tasks::SASTask {
    let task = simplify::trivial_task(solvable);
    py_sas_tasks::from_internal(&task)
}

pub fn solvable_sas_task(_msg: &str) -> py_sas_tasks::SASTask {
    let task = simplify::trivial_task(true);
    py_sas_tasks::from_internal(&task)
}

pub fn unsolvable_sas_task(_msg: &str) -> py_sas_tasks::SASTask {
    let task = simplify::trivial_task(false);
    py_sas_tasks::from_internal(&task)
}

pub fn pddl_to_sas(dom: &pddl::Domain, prob: &pddl::Problem) -> Result<py_sas_tasks::SASTask, String> {
    translate_from_ast(dom, prob, &TranslateConfig::default())
}

pub fn translate_task(
    strips_to_sas: &mut HashMap<Literal, Vec<(usize, usize)>>,
    ranges: &mut Vec<usize>,
    translation_key: &mut Vec<Vec<String>>,
    numeric_strips_to_sas: &HashMap<String, usize>,
    num_count: usize,
    mutex_dict: &mut HashMap<Literal, Vec<(usize, usize)>>,
    mutex_ranges: &mut Vec<usize>,
    mutex_key: &[Vec<(usize, usize)>],
    init: &[Literal],
    num_init: &HashMap<String, f64>,
    goal_list: &[Literal],
    global_constraint: &Literal,
    actions: &[StripsOperator],
    axioms: &[StripsAxiom],
    num_axioms: &[InstantiatedNumericAxiom],
    _num_axioms_by_layer: &HashMap<i32, Vec<InstantiatedNumericAxiom>>,
    _num_axiom_map: &HashMap<PrimitiveNumericExpression, InstantiatedNumericAxiom>,
    _const_num_axioms: &[InstantiatedNumericAxiom],
    metric: (String, isize),
    implied_facts: &HashMap<(usize, usize), Vec<(usize, usize)>>,
    _init_constant_predicates: &[Literal],
    _init_constant_numerics: &HashMap<String, f64>,
) -> InternalSASTask {
    let mut init_values: Vec<i32> = ranges.iter().map(|r| (r - 1) as i32).collect();
    for fact in init {
        if let Some(pairs) = strips_to_sas.get(fact) {
            for (var, val) in pairs {
                init_values[*var] = *val as i32;
            }
        }
    }

    let mut comp_axiom_dict: HashMap<(String, Vec<usize>), Literal> = HashMap::new();
    let mut sas_comp_axioms: Vec<CompareAxiom> = Vec::new();

    let goal_dict_list = translate_strips_conditions(
        goal_list,
        strips_to_sas,
        ranges,
        numeric_strips_to_sas,
        mutex_dict,
        mutex_ranges,
        &mut comp_axiom_dict,
        &mut sas_comp_axioms,
    );
    if goal_dict_list.is_none() {
        return unsolvable_sas_task_internal();
    }
    let goal_dict_list = goal_dict_list.unwrap();
    let goal_pairs: Vec<(usize, usize)> = goal_dict_list
        .get(0)
        .map(|d| d.iter().map(|(k, v)| (*k, *v)).collect())
        .unwrap_or_default();
    if goal_pairs.is_empty() {
        return solvable_sas_task_internal();
    }

    let global_constraint_dict_list = translate_strips_conditions(
        std::slice::from_ref(global_constraint),
        strips_to_sas,
        ranges,
        numeric_strips_to_sas,
        mutex_dict,
        mutex_ranges,
        &mut comp_axiom_dict,
        &mut sas_comp_axioms,
    )
    .unwrap_or_default();
    let global_constraint_pair = global_constraint_dict_list
        .get(0)
        .and_then(|d| d.iter().next())
        .map(|(k, v)| (*k, *v));

    add_key_to_comp_axioms(&mut sas_comp_axioms, translation_key);

    let mut num_init_values = vec![0.0; num_count];
    for (name, value) in num_init {
        if let Some(idx) = numeric_strips_to_sas.get(name) {
            if *idx < num_init_values.len() {
                num_init_values[*idx] = *value;
            }
        }
    }

    let relevant_numeric_vars: HashSet<usize> = numeric_strips_to_sas.values().copied().collect();

    let operators = translate_strips_operators(
        actions,
        strips_to_sas,
        ranges,
        numeric_strips_to_sas,
        mutex_dict,
        mutex_ranges,
        implied_facts,
        &mut comp_axiom_dict,
        &mut sas_comp_axioms,
        num_count,
        &relevant_numeric_vars,
    );
    let axioms_out = translate_strips_axioms(
        axioms,
        strips_to_sas,
        ranges,
        numeric_strips_to_sas,
        mutex_dict,
        mutex_ranges,
        &mut comp_axiom_dict,
        &mut sas_comp_axioms,
    );

    let numeric_axioms: Vec<crate::translate::sas::NumericAxiom> = num_axioms
        .iter()
        .filter_map(|ax| translate_numeric_axiom(ax, strips_to_sas, numeric_strips_to_sas))
        .collect();

    let variables = translation_key
        .iter()
        .map(|values| crate::translate::sas::Variable {
            value_names: values.clone(),
        })
        .collect();

    InternalSASTask {
        variables,
        operators,
        numeric_variables: Vec::new(),
        numeric_axioms,
        comparison_axioms: sas_comp_axioms
            .iter()
            .map(|ax| crate::translate::sas::CompareAxiom {
                comp: ax.comp.clone(),
                parts: ax.parts.clone(),
                effect_var: ax.effect_var,
            })
            .collect(),
        axioms: axioms_out,
        numeric_init: num_init_values,
        mutex_groups: mutex_key.to_vec(),
        ranges: ranges.clone(),
        axiom_layers: vec![-1; ranges.len()],
        init: init_values,
        goal: goal_pairs,
        translation_key: translation_key.clone(),
        canonical_variables: Vec::new(),
        canonical_operators: Vec::new(),
        canonical_metric: None,
        metric: metric,
        global_constraint: global_constraint_pair,
        comp_axiom_layer: -1,
    }
}
