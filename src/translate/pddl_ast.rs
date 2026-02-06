use crate::translate::pddl_parser::SExpr;

const KW_DEFINE: &str = "define";
const KW_DOMAIN: &str = "domain";
const KW_PROBLEM: &str = "problem";
const KW_PREDICATES: &str = ":predicates";
const KW_FUNCTIONS: &str = ":functions";
const KW_TYPES: &str = ":types";
const KW_ACTION: &str = ":action";
const KW_PARAMETERS: &str = ":parameters";
const KW_PRECONDITION: &str = ":precondition";
const KW_EFFECT: &str = ":effect";
const KW_OBJECTS: &str = ":objects";
const KW_INIT: &str = ":init";
const KW_GOAL: &str = ":goal";
const KW_METRIC: &str = ":metric";
const KW_AND: &str = "and";
const KW_NOT: &str = "not";
const KW_INCREASE: &str = "increase";
const KW_DECREASE: &str = "decrease";
const KW_EQUAL: &str = "=";
const TYPE_MARKER: &str = "-";

#[derive(Debug, Clone)]
pub struct Domain {
    pub name: String,
    /// predicates: name -> list of (param, type)
    pub predicates: Vec<(String, Vec<(String, Option<String>)>)>,
    /// functions (numeric fluents): name -> params
    pub functions: Vec<(String, Vec<(String, Option<String>)>)>,
    /// types: (type_name, supertype)
    pub types: Vec<(String, Option<String>)>,
    pub actions: Vec<Action>,
}

#[derive(Debug, Clone)]
pub struct Problem {
    pub name: String,
    pub objects: Vec<(String, Option<String>)>,
    pub init: Vec<SExpr>,
    pub goal: Option<SExpr>,
    pub metric: Option<(String, SExpr)>,
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

    /// Check if this condition contains an existential quantifier anywhere in the tree
    pub fn has_existential_part(&self) -> bool {
        match self {
            Condition::Exists(_, _) => true,
            Condition::Not(c) => c.has_existential_part(),
            Condition::And(parts) | Condition::Or(parts) => {
                parts.iter().any(|p| p.has_existential_part())
            }
            Condition::Forall(_, c) => c.has_existential_part(),
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
            Condition::And(parts) => Condition::Or(parts.iter().map(|p| p.negate()).collect()),
            Condition::Or(parts) => Condition::And(parts.iter().map(|p| p.negate()).collect()),
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

#[allow(dead_code)]
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
    fn parse_atom(name: &str, args: &[SExpr]) -> Condition {
        let parsed_args = args
            .iter()
            .filter_map(|a| {
                if let SExpr::Atom(arg) = a {
                    Some(arg.clone())
                } else {
                    None
                }
            })
            .collect();
        Condition::Atom(name.to_string(), parsed_args)
    }

    fn build_compare_to_zero(op: &str, left: &SExpr, right: &SExpr) -> Condition {
        let difference = SExpr::List(vec![
            SExpr::Atom("-".to_string()),
            left.clone(),
            right.clone(),
        ]);
        let zero = SExpr::Atom("0".to_string());
        Condition::Comparison(op.to_string(), difference, zero)
    }

    match s {
        SExpr::Atom(a) => Condition::Atom(a.clone(), vec![]),
        SExpr::List(list) => {
            if list.is_empty() {
                return Condition::True;
            }
            if let SExpr::Atom(k) = &list[0] {
                let key = k.to_lowercase();
                match key.as_str() {
                    KW_AND => Condition::And(list[1..].iter().map(sexpr_to_condition).collect()),
                    KW_NOT => list
                        .get(1)
                        .map(|inner| Condition::Not(Box::new(sexpr_to_condition(inner))))
                        .unwrap_or(Condition::True),
                    "<=" | ">=" | "<" | ">" | KW_EQUAL => list
                        .get(1)
                        .and_then(|left| {
                            list.get(2)
                                .map(|right| build_compare_to_zero(k, left, right))
                        })
                        .unwrap_or(Condition::True),
                    _ => parse_atom(k, &list[1..]),
                }
            } else {
                Condition::True
            }
        }
    }
}

pub fn sexpr_to_effect(s: &SExpr) -> Effect {
    fn parse_name_and_args(list: &[SExpr]) -> Option<(String, Vec<String>)> {
        let name = match list.get(0) {
            Some(SExpr::Atom(n)) => n.clone(),
            _ => return None,
        };
        let args = list[1..]
            .iter()
            .filter_map(|x| match x {
                SExpr::Atom(a) => Some(a.clone()),
                _ => None,
            })
            .collect();
        Some((name, args))
    }

    fn parse_numeric_effect(list: &[SExpr], kind: &str) -> Option<Effect> {
        if list.len() < 3 {
            return None;
        }
        let (func, args) = match &list[1] {
            SExpr::List(inner) => {
                let name = match inner.get(0) {
                    Some(SExpr::Atom(name)) => name.clone(),
                    _ => return None,
                };
                let args = inner[1..]
                    .iter()
                    .filter_map(|x| match x {
                        SExpr::Atom(a) => Some(a.clone()),
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                (name, args)
            }
            _ => return None,
        };
        let value = match &list[2] {
            SExpr::Atom(v) => v.parse::<i64>().ok()?,
            _ => return None,
        };
        match kind {
            KW_INCREASE => Some(Effect::Increase(func, args, value)),
            KW_DECREASE => Some(Effect::Decrease(func, args, value)),
            _ => None,
        }
    }

    match s {
        SExpr::Atom(a) => Effect::Add(a.clone(), vec![]),
        SExpr::List(list) => {
            if list.is_empty() {
                return Effect::And(vec![]);
            }
            if let SExpr::Atom(k) = &list[0] {
                let key = k.to_lowercase();
                match key.as_str() {
                    KW_AND => Effect::And(list[1..].iter().map(sexpr_to_effect).collect()),
                    KW_NOT => list
                        .get(1)
                        .and_then(|inner| match inner {
                            SExpr::List(items) => parse_name_and_args(items)
                                .map(|(name, args)| Effect::Del(name, args)),
                            _ => None,
                        })
                        .unwrap_or(Effect::And(vec![])),
                    KW_INCREASE | KW_DECREASE => {
                        parse_numeric_effect(&list, key.as_str()).unwrap_or(Effect::And(vec![]))
                    }
                    _ => parse_name_and_args(list)
                        .map(|(name, args)| Effect::Add(name, args))
                        .unwrap_or(Effect::And(vec![])),
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
            SExpr::Atom(a) if a == TYPE_MARKER => {
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
            _ => i += 1,
        }
    }
    for name in buffer.drain(..) {
        out.push((name, None));
    }
    out
}

impl Domain {
    pub fn from_sexprs(forms: &[SExpr]) -> Option<Self> {
        forms.iter().find_map(parse_domain_form)
    }
}

fn parse_domain_form(form: &SExpr) -> Option<Domain> {
    let items = match form {
        SExpr::List(items) => items,
        _ => return None,
    };
    if items.len() < 2 {
        return None;
    }
    let keyword = match &items[0] {
        SExpr::Atom(a) if a.to_lowercase() == KW_DEFINE => a,
        _ => return None,
    };
    let _ = keyword;
    let domain_spec = match &items[1] {
        SExpr::List(domain_spec) if domain_spec.len() >= 2 => domain_spec,
        _ => return None,
    };
    let domain_kw = match &domain_spec[0] {
        SExpr::Atom(k) if k.to_lowercase() == KW_DOMAIN => k,
        _ => return None,
    };
    let _ = domain_kw;
    let name = match &domain_spec[1] {
        SExpr::Atom(n) => n.clone(),
        _ => String::new(),
    };

    let mut predicates = Vec::new();
    let mut functions = Vec::new();
    let mut types = Vec::new();
    let mut actions = Vec::new();

    for section in &items[2..] {
        let list = match section {
            SExpr::List(list) if !list.is_empty() => list,
            _ => continue,
        };
        let key = match &list[0] {
            SExpr::Atom(k) => k.to_lowercase(),
            _ => continue,
        };
        let content = &list[1..];
        match key.as_str() {
            KW_TYPES | "types" => {
                types.extend(parse_typed_list(content));
            }
            KW_PREDICATES | "predicates" => parse_predicates(content, &mut predicates),
            KW_FUNCTIONS | "functions" => parse_functions(content, &mut functions),
            KW_ACTION | "action" => {
                if let Some(act) = Action::from_section(content) {
                    actions.push(act);
                }
            }
            _ => {}
        }
    }

    Some(Domain {
        name,
        predicates,
        functions,
        types,
        actions,
    })
}

fn parse_predicates(
    content: &[SExpr],
    predicates: &mut Vec<(String, Vec<(String, Option<String>)>)>,
) {
    for item in content {
        if let SExpr::List(p) = item {
            if let Some(SExpr::Atom(nm)) = p.get(0) {
                let params = parse_typed_list(&p[1..]);
                predicates.push((nm.clone(), params));
            }
        }
    }
}

fn parse_functions(
    content: &[SExpr],
    functions: &mut Vec<(String, Vec<(String, Option<String>)>)>,
) {
    for item in content {
        if let SExpr::List(p) = item {
            if let Some(SExpr::Atom(nm)) = p.get(0) {
                let params = parse_typed_list(&p[1..]);
                functions.push((nm.clone(), params));
            }
        }
    }
}

impl Action {
    fn from_section(content: &[SExpr]) -> Option<Self> {
        // content normally: (name) :parameters (...) :precondition (...) :effect (...)
        let name = match content.get(0) {
            Some(SExpr::Atom(a)) => a.clone(),
            _ => String::new(),
        };
        let mut params = Vec::new();
        let mut pre = None;
        let mut eff = None;

        let mut i = 1;
        while i < content.len() {
            let key = match &content[i] {
                SExpr::Atom(k) => k.to_lowercase(),
                _ => {
                    i += 1;
                    continue;
                }
            };
            match key.as_str() {
                KW_PARAMETERS if i + 1 < content.len() => {
                    if let SExpr::List(list) = &content[i + 1] {
                        params = parse_typed_list(list);
                    }
                    i += 2;
                }
                KW_PRECONDITION if i + 1 < content.len() => {
                    pre = Some(content[i + 1].clone());
                    i += 2;
                }
                KW_EFFECT if i + 1 < content.len() => {
                    eff = Some(content[i + 1].clone());
                    i += 2;
                }
                _ => i += 1,
            }
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
        forms.iter().find_map(parse_problem_form)
    }
}

fn parse_problem_form(form: &SExpr) -> Option<Problem> {
    let items = match form {
        SExpr::List(items) => items,
        _ => return None,
    };
    if items.len() < 2 {
        return None;
    }
    let key = match &items[0] {
        SExpr::Atom(a) if a.to_lowercase() == KW_DEFINE => a,
        _ => return None,
    };
    let _ = key;
    let mut name = String::new();
    let mut objects = Vec::new();
    let mut init = Vec::new();
    let mut goal = None;
    let mut metric = None;

    for part in &items[1..] {
        let inner = match part {
            SExpr::List(inner) if !inner.is_empty() => inner,
            _ => continue,
        };
        let section = match &inner[0] {
            SExpr::Atom(atom0) => atom0.to_lowercase(),
            _ => continue,
        };
        match section.as_str() {
            KW_PROBLEM => {
                if let Some(SExpr::Atom(n)) = inner.get(1) {
                    name = n.clone();
                }
            }
            KW_OBJECTS | "objects" => {
                objects.extend(parse_typed_list(&inner[1..]));
            }
            KW_INIT | "init" => {
                init.extend(inner[1..].iter().cloned());
            }
            KW_GOAL | "goal" => {
                if let Some(goal_expr) = inner.get(1) {
                    goal = Some(goal_expr.clone());
                }
            }
            KW_METRIC | "metric" => {
                if inner.len() >= 3 {
                    if let SExpr::Atom(direction) = &inner[1] {
                        let dir = match direction.to_lowercase().as_str() {
                            "minimize" => "<".to_string(),
                            "maximize" => ">".to_string(),
                            _ => direction.clone(),
                        };
                        metric = inner.get(2).cloned().map(|expr| (dir, expr));
                    }
                }
            }
            _ => {}
        }
    }

    Some(Problem {
        name,
        objects,
        init,
        goal,
        metric,
    })
}

#[cfg(test)]
mod instantiate_tests {
    use super::*;
    use crate::translate::instantiate;
    use crate::translate::normalize;

    #[test]
    fn ground_smoke() {
        let task = crate::translate::pddl::PddlTask::from_files(
            std::path::Path::new("misc/plant-watering/domain.pddl"),
            std::path::Path::new("misc/plant-watering/prob_4_1_1.pddl"),
        )
        .unwrap();
        let dom = Domain::from_sexprs(&task.domain_forms).expect("domain parsed");
        let prob = Problem::from_sexprs(&task.problem_forms).expect("problem parsed");
        let mut norm_task = normalize::NormalizableTask::from_ast(&dom, &prob);
        normalize::normalize(&mut norm_task).expect("normalization failed");
        let result = instantiate::explore_normalized(&norm_task).expect("instantiation failed");
        // grounding may produce many ops; check it's non-empty
        if result.grounded_ops.is_empty() {
            panic!(
                "grounding produced zero ops; domain.actions={}, problem.objects={}",
                dom.actions.len(),
                prob.objects.len()
            );
        }
    }
}
