use crate::translate::pddl_parser::SExpr;

#[derive(Debug, Clone)]
pub struct Domain {
    pub name: String,
    /// predicates: name -> list of (param, type)
    pub predicates: Vec<(String, Vec<(String, Option<String>)>)>,
    /// functions (numeric fluents): name -> params
    pub functions: Vec<(String, Vec<(String, Option<String>)>)>,
    pub actions: Vec<Action>,
}

#[derive(Debug, Clone)]
pub struct Problem {
    pub name: String,
    pub objects: Vec<(String, Option<String>)>,
    pub init: Vec<SExpr>,
    pub goal: Option<SExpr>,
}

#[derive(Debug, Clone)]
pub struct Action {
    pub name: String,
    pub parameters: Vec<(String, Option<String>)>,
    pub precond: Option<SExpr>,
    pub effect: Option<SExpr>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Condition {
    Atom(String, Vec<String>), // predicate name and args
    Not(Box<Condition>),
    And(Vec<Condition>),
    Or(Vec<Condition>),
    Forall(Vec<(String, Option<String>)>, Box<Condition>), // parameters and condition
    Exists(Vec<(String, Option<String>)>, Box<Condition>), // parameters and condition
    Comparison(String, SExpr, SExpr), // op, left, right as raw SExprs (not Hash-able)
    True,
}

impl Condition {
    /// Check if this condition contains a universal quantifier anywhere in the tree
    pub fn has_universal_part(&self) -> bool {
        match self {
            Condition::Forall(_, _) => true,
            Condition::Not(c) => c.has_universal_part(),
            Condition::And(parts) | Condition::Or(parts) => {
                parts.iter().any(|p| p.has_universal_part())
            }
            Condition::Exists(_, c) => c.has_universal_part(),
            _ => false,
        }
    }

    /// Check if this condition contains a disjunction anywhere in the tree
    pub fn has_disjunction(&self) -> bool {
        match self {
            Condition::Or(_) => true,
            Condition::Not(c) => c.has_disjunction(),
            Condition::And(parts) => parts.iter().any(|p| p.has_disjunction()),
            Condition::Exists(_, c) | Condition::Forall(_, c) => c.has_disjunction(),
            _ => false,
        }
    }

    /// Collect all free variables (variables not bound by quantifiers)
    pub fn free_variables(&self) -> std::collections::HashSet<String> {
        use std::collections::HashSet;
        match self {
            Condition::Atom(_, args) => args
                .iter()
                .filter(|a| a.starts_with('?'))
                .cloned()
                .collect(),
            Condition::Not(c) => c.free_variables(),
            Condition::And(parts) | Condition::Or(parts) => {
                parts.iter().flat_map(|p| p.free_variables()).collect()
            }
            Condition::Forall(params, c) | Condition::Exists(params, c) => {
                let mut vars = c.free_variables();
                for (name, _) in params {
                    vars.remove(name);
                }
                vars
            }
            Condition::Comparison(_, _, _) => HashSet::new(), // simplified
            Condition::True => HashSet::new(),
        }
    }

    /// Negate this condition (De Morgan's laws + quantifier duality)
    pub fn negate(&self) -> Condition {
        match self {
            Condition::Atom(_, _) => Condition::Not(Box::new(self.clone())),
            Condition::Not(c) => (**c).clone(),
            Condition::And(parts) => {
                Condition::Or(parts.iter().map(|p| p.negate()).collect())
            }
            Condition::Or(parts) => {
                Condition::And(parts.iter().map(|p| p.negate()).collect())
            }
            Condition::Forall(params, c) => {
                // ¬∀x.φ ≡ ∃x.¬φ
                Condition::Exists(params.clone(), Box::new(c.negate()))
            }
            Condition::Exists(params, c) => {
                // ¬∃x.φ ≡ ∀x.¬φ
                Condition::Forall(params.clone(), Box::new(c.negate()))
            }
            Condition::Comparison(_, _, _) => {
                // For simplicity, wrap in Not
                Condition::Not(Box::new(self.clone()))
            }
            Condition::True => Condition::Not(Box::new(Condition::True)),
        }
    }

    /// Replace the sub-parts of this condition with new parts
    pub fn change_parts(&self, new_parts: Vec<Condition>) -> Condition {
        match self {
            Condition::And(_) => Condition::And(new_parts),
            Condition::Or(_) => Condition::Or(new_parts),
            Condition::Not(_) => {
                if new_parts.len() == 1 {
                    Condition::Not(Box::new(new_parts[0].clone()))
                } else {
                    self.clone()
                }
            }
            Condition::Forall(params, _) => {
                if new_parts.len() == 1 {
                    Condition::Forall(params.clone(), Box::new(new_parts[0].clone()))
                } else {
                    self.clone()
                }
            }
            Condition::Exists(params, _) => {
                if new_parts.len() == 1 {
                    Condition::Exists(params.clone(), Box::new(new_parts[0].clone()))
                } else {
                    self.clone()
                }
            }
            _ => self.clone(),
        }
    }

    /// Get sub-conditions for recursive processing
    pub fn parts(&self) -> Vec<Condition> {
        match self {
            Condition::Not(c) => vec![(**c).clone()],
            Condition::And(parts) | Condition::Or(parts) => parts.clone(),
            Condition::Forall(_, c) | Condition::Exists(_, c) => vec![(**c).clone()],
            _ => vec![],
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Effect {
    Add(String, Vec<String>),
    Del(String, Vec<String>),
    Increase(String, Vec<String>, i64),
    Decrease(String, Vec<String>, i64),
    And(Vec<Effect>),
}

fn sexpr_to_string(s: &SExpr) -> String {
    match s {
        SExpr::Atom(a) => a.clone(),
        SExpr::List(list) => {
            let parts: Vec<String> = list.iter().map(|p| sexpr_to_string(p)).collect();
            format!("({})", parts.join(" "))
        }
    }
}

pub fn sexpr_to_condition(s: &SExpr) -> Condition {
    match s {
        SExpr::Atom(a) => Condition::Atom(a.clone(), vec![]),
        SExpr::List(list) => {
            if list.is_empty() {
                return Condition::True;
            }
            if let SExpr::Atom(k) = &list[0] {
                match k.to_lowercase().as_str() {
                    "and" => {
                        let mut cs = Vec::new();
                        for item in &list[1..] {
                            cs.push(sexpr_to_condition(item));
                        }
                        Condition::And(cs)
                    }
                    "not" => {
                        if list.len() >= 2 {
                            Condition::Not(Box::new(sexpr_to_condition(&list[1])))
                        } else {
                            Condition::True
                        }
                    }
                    "<=" | ">=" | "<" | ">" | "=" => {
                        if list.len() >= 3 {
                            // store raw SExprs for richer parsing later
                            return Condition::Comparison(
                                k.clone(),
                                list[1].clone(),
                                list[2].clone(),
                            );
                        } else {
                            return Condition::True;
                        }
                    }
                    _ => {
                        // atom-like: predicate name followed by args
                        let name = k.clone();
                        let mut args = Vec::new();
                        for a in &list[1..] {
                            if let SExpr::Atom(arg) = a {
                                args.push(arg.clone());
                            }
                        }
                        Condition::Atom(name, args)
                    }
                }
            } else {
                Condition::True
            }
        }
    }
}

pub fn sexpr_to_effect(s: &SExpr) -> Effect {
    match s {
        SExpr::Atom(a) => Effect::Add(a.clone(), vec![]),
        SExpr::List(list) => {
            if list.is_empty() {
                return Effect::And(vec![]);
            }
            if let SExpr::Atom(k) = &list[0] {
                match k.to_lowercase().as_str() {
                    "and" => {
                        let mut es = Vec::new();
                        for item in &list[1..] {
                            es.push(sexpr_to_effect(item));
                        }
                        Effect::And(es)
                    }
                    "not" => {
                        // not (atom) interpreted as delete
                        if list.len() >= 2 {
                            if let SExpr::List(inner) = &list[1] {
                                if let SExpr::Atom(name) = &inner[0] {
                                    let args = inner[1..]
                                        .iter()
                                        .filter_map(|x| match x {
                                            SExpr::Atom(a) => Some(a.clone()),
                                            _ => None,
                                        })
                                        .collect();
                                    return Effect::Del(name.clone(), args);
                                }
                            }
                        }
                        Effect::And(vec![])
                    }
                    "increase" => {
                        // (increase (f) val)
                        if list.len() >= 3 {
                            let func = if let SExpr::List(inner) = &list[1] {
                                if let SExpr::Atom(name) = &inner[0] {
                                    name.clone()
                                } else {
                                    "".to_string()
                                }
                            } else {
                                "".to_string()
                            };
                            if let SExpr::Atom(v) = &list[2] {
                                if let Ok(n) = v.parse::<i64>() {
                                    return Effect::Increase(func, vec![], n);
                                }
                            }
                        }
                        Effect::And(vec![])
                    }
                    "decrease" => {
                        if list.len() >= 3 {
                            let func = if let SExpr::List(inner) = &list[1] {
                                if let SExpr::Atom(name) = &inner[0] {
                                    name.clone()
                                } else {
                                    "".to_string()
                                }
                            } else {
                                "".to_string()
                            };
                            if let SExpr::Atom(v) = &list[2] {
                                if let Ok(n) = v.parse::<i64>() {
                                    return Effect::Decrease(func, vec![], n);
                                }
                            }
                        }
                        Effect::And(vec![])
                    }
                    _ => {
                        // treat as add
                        let name = k.clone();
                        let args = list[1..]
                            .iter()
                            .filter_map(|x| match x {
                                SExpr::Atom(a) => Some(a.clone()),
                                _ => None,
                            })
                            .collect();
                        Effect::Add(name, args)
                    }
                }
            } else {
                Effect::And(vec![])
            }
        }
    }
}

fn parse_typed_list(list: &[SExpr]) -> Vec<(String, Option<String>)> {
    // parse sequences like a b c - type d e - other
    let mut out = Vec::new();
    let mut buffer: Vec<String> = Vec::new();
    let mut i = 0;
    while i < list.len() {
        match &list[i] {
            SExpr::Atom(a) if a == "-" => {
                // next token is type
                if i + 1 < list.len() {
                    if let SExpr::Atom(t) = &list[i + 1] {
                        for name in buffer.drain(..) {
                            out.push((name, Some(t.clone())));
                        }
                        i += 2;
                        continue;
                    }
                }
                i += 1;
            }
            SExpr::Atom(a) => {
                buffer.push(a.clone());
                i += 1;
            }
            _ => {
                i += 1;
            }
        }
    }
    for name in buffer.drain(..) {
        out.push((name, None));
    }
    out
}

impl Domain {
    pub fn from_sexprs(forms: &[SExpr]) -> Option<Self> {
        for f in forms {
            if let SExpr::List(items) = f {
                if items.len() >= 2 {
                    if let SExpr::Atom(a) = &items[0] {
                        if a.to_lowercase() == "define" {
                            // second element usually (domain NAME)
                            if items.len() >= 2 {
                                if let SExpr::List(domain_spec) = &items[1] {
                                    if domain_spec.len() >= 2 {
                                        if let SExpr::Atom(domain_kw) = &domain_spec[0] {
                                            if domain_kw.to_lowercase() == "domain" {
                                                let name = match &domain_spec[1] {
                                                    SExpr::Atom(n) => n.clone(),
                                                    _ => "".to_string(),
                                                };
                                                // collect sections and actions from the rest
                                                let mut predicates = Vec::new();
                                                let mut functions = Vec::new();
                                                let mut actions = Vec::new();
                                                for section in &items[2..] {
                                                    match section {
                                                        SExpr::List(list) => {
                                                            if !list.is_empty() {
                                                                if let SExpr::Atom(k) = &list[0] {
                                                                    let key = k.clone();
                                                                    let content =
                                                                        list[1..].to_vec();
                                                                    match key
                                                                        .to_lowercase()
                                                                        .as_str()
                                                                    {
                                                                        ":predicates"
                                                                        | "predicates" => {
                                                                            // each content item may be a predicate list
                                                                            for item in &content {
                                                                                if let SExpr::List(
                                                                                    p,
                                                                                ) = item
                                                                                {
                                                                                    if !p.is_empty()
                                                                                    {
                                                                                        if let SExpr::Atom(nm) = &p[0] {
                                                                                            let params = parse_typed_list(&p[1..]);
                                                                                            predicates.push((nm.clone(), params));
                                                                                        }
                                                                                    }
                                                                                }
                                                                            }
                                                                        }
                                                                        ":functions"
                                                                        | "functions" => {
                                                                            for item in &content {
                                                                                if let SExpr::List(
                                                                                    p,
                                                                                ) = item
                                                                                {
                                                                                    if !p.is_empty()
                                                                                    {
                                                                                        if let SExpr::Atom(nm) = &p[0] {
                                                                                            let params = parse_typed_list(&p[1..]);
                                                                                            functions.push((nm.clone(), params));
                                                                                        }
                                                                                    }
                                                                                }
                                                                            }
                                                                        }
                                                                        ":action" | "action" => {
                                                                            if let Some(act) =
                                                                                Action::from_section(
                                                                                    &content,
                                                                                )
                                                                            {
                                                                                actions.push(act);
                                                                            }
                                                                        }
                                                                        _ => {}
                                                                    }
                                                                }
                                                            }
                                                        }
                                                        _ => {}
                                                    }
                                                }
                                                return Some(Domain {
                                                    name,
                                                    predicates,
                                                    functions,
                                                    actions,
                                                });
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        None
    }
}

impl Action {
    fn from_section(content: &[SExpr]) -> Option<Self> {
        // content normally: (name) :parameters (...) :precondition (...) :effect (...)
        let mut name = "".to_string();
        let mut params = Vec::new();
        let mut pre = None;
        let mut eff = None;
        // First element might be the action name as Atom
        if !content.is_empty() {
            if let SExpr::Atom(a) = &content[0] {
                name = a.clone();
            }
        }
        // scan for keywords
        let mut i = 1;
        while i < content.len() {
            match &content[i] {
                SExpr::Atom(k) => {
                    let key = k.to_lowercase();
                    if key == ":parameters" && i + 1 < content.len() {
                        if let SExpr::List(list) = &content[i + 1] {
                            // parameters are atoms and typed segments; keep raw atom names
                            let mut j = 0;
                            while j < list.len() {
                                if let SExpr::Atom(p) = &list[j] {
                                    // lookahead for type marker '-'
                                    if j + 2 < list.len() {
                                        if let SExpr::Atom(dash) = &list[j + 1] {
                                            if dash == "-" {
                                                if let SExpr::Atom(t) = &list[j + 2] {
                                                    params.push((p.clone(), Some(t.clone())));
                                                    j += 3;
                                                    continue;
                                                }
                                            }
                                        }
                                    }
                                    params.push((p.clone(), None));
                                }
                                j += 1;
                            }
                        }
                        i += 2;
                        continue;
                    } else if key == ":precondition" && i + 1 < content.len() {
                        pre = Some(content[i + 1].clone());
                        i += 2;
                        continue;
                    } else if key == ":effect" && i + 1 < content.len() {
                        eff = Some(content[i + 1].clone());
                        i += 2;
                        continue;
                    }
                }
                _ => {}
            }
            i += 1;
        }
        Some(Action {
            name,
            parameters: params,
            precond: pre,
            effect: eff,
        })
    }
}

// Substitute variable tokens like "?x" in a Condition according to mapping
pub fn substitute_condition(
    cond: &Condition,
    mapping: &std::collections::HashMap<String, String>,
) -> Condition {
    match cond {
        Condition::Atom(name, args) => {
            let args2 = args
                .iter()
                .map(|a| mapping.get(a).cloned().unwrap_or_else(|| a.clone()))
                .collect();
            Condition::Atom(name.clone(), args2)
        }
        Condition::Not(c) => Condition::Not(Box::new(substitute_condition(c, mapping))),
        Condition::And(v) => {
            Condition::And(v.iter().map(|c| substitute_condition(c, mapping)).collect())
        }
        Condition::Or(v) => {
            Condition::Or(v.iter().map(|c| substitute_condition(c, mapping)).collect())
        }
        Condition::Forall(params, c) => {
            Condition::Forall(params.clone(), Box::new(substitute_condition(c, mapping)))
        }
        Condition::Exists(params, c) => {
            Condition::Exists(params.clone(), Box::new(substitute_condition(c, mapping)))
        }
        Condition::Comparison(op, l, r) => {
            // comparisons store raw SExprs; substitute variable tokens inside them
            fn substitute_sexpr(
                sex: &SExpr,
                mapping: &std::collections::HashMap<String, String>,
            ) -> SExpr {
                match sex {
                    SExpr::Atom(a) => {
                        if a.starts_with('?') {
                            if let Some(v) = mapping.get(a) {
                                SExpr::Atom(v.clone())
                            } else {
                                SExpr::Atom(a.clone())
                            }
                        } else {
                            SExpr::Atom(a.clone())
                        }
                    }
                    SExpr::List(list) => {
                        SExpr::List(list.iter().map(|s| substitute_sexpr(s, mapping)).collect())
                    }
                }
            }
            Condition::Comparison(
                op.clone(),
                substitute_sexpr(l, mapping),
                substitute_sexpr(r, mapping),
            )
        }
        Condition::True => Condition::True,
    }
}

pub fn substitute_effect(
    eff: &Effect,
    mapping: &std::collections::HashMap<String, String>,
) -> Effect {
    match eff {
        Effect::Add(n, args) => Effect::Add(
            n.clone(),
            args.iter()
                .map(|a| mapping.get(a).cloned().unwrap_or_else(|| a.clone()))
                .collect(),
        ),
        Effect::Del(n, args) => Effect::Del(
            n.clone(),
            args.iter()
                .map(|a| mapping.get(a).cloned().unwrap_or_else(|| a.clone()))
                .collect(),
        ),
        Effect::Increase(n, args, v) => Effect::Increase(
            n.clone(),
            args.iter()
                .map(|a| mapping.get(a).cloned().unwrap_or_else(|| a.clone()))
                .collect(),
            *v,
        ),
        Effect::Decrease(n, args, v) => Effect::Decrease(
            n.clone(),
            args.iter()
                .map(|a| mapping.get(a).cloned().unwrap_or_else(|| a.clone()))
                .collect(),
            *v,
        ),
        Effect::And(v) => Effect::And(v.iter().map(|e| substitute_effect(e, mapping)).collect()),
    }
}

impl Problem {
    pub fn from_sexprs(forms: &[SExpr]) -> Option<Self> {
        for f in forms {
            if let SExpr::List(items) = f {
                if items.len() >= 2 {
                    if let SExpr::Atom(a) = &items[0] {
                        if a.to_lowercase() == "define" {
                            // find (:problem NAME) or (:problem ...)
                            let mut name = String::new();
                            let mut objects = Vec::new();
                            let mut init = Vec::new();
                            let mut goal = None;
                            for part in &items[1..] {
                                match part {
                                    SExpr::List(inner) => {
                                        if inner.is_empty() {
                                            continue;
                                        }
                                        if let SExpr::Atom(atom0) = &inner[0] {
                                            match atom0.to_lowercase().as_str() {
                                                ":problem" | "problem" => {
                                                    if inner.len() >= 2 {
                                                        if let SExpr::Atom(n) = &inner[1] {
                                                            name = n.clone();
                                                        }
                                                    }
                                                }
                                                ":objects" | "objects" => {
                                                    // parse typed object lists like a b c - type
                                                    let typed = parse_typed_list(&inner[1..]);
                                                    for (name, tp) in typed {
                                                        objects.push((name, tp));
                                                    }
                                                }
                                                ":init" | "init" => {
                                                    for token in &inner[1..] {
                                                        init.push(token.clone());
                                                    }
                                                }
                                                ":goal" | "goal" => {
                                                    if inner.len() >= 2 {
                                                        goal = Some(inner[1].clone());
                                                    }
                                                }
                                                _ => {}
                                            }
                                        }
                                    }
                                    _ => {}
                                }
                            }
                            return Some(Problem {
                                name,
                                objects,
                                init,
                                goal,
                            });
                        }
                    }
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod instantiate_tests {
    use super::*;
    use crate::translate::instantiate;

    #[test]
    fn ground_smoke() {
        let task = crate::translate::pddl::PddlTask::from_files(
            std::path::Path::new("pddl/domain.pddl"),
            std::path::Path::new("pddl/pfile1.pddl"),
        )
        .unwrap();
        let dom = Domain::from_sexprs(&task.domain_forms).expect("domain parsed");
        let prob = Problem::from_sexprs(&task.problem_forms).expect("problem parsed");
        let ops = instantiate::ground(&dom, &prob);
        // grounding may produce many ops; check it's non-empty
        if ops.is_empty() {
            panic!(
                "grounding produced zero ops; domain.actions={}, problem.objects={}",
                dom.actions.len(),
                prob.objects.len()
            );
        }
    }
}
