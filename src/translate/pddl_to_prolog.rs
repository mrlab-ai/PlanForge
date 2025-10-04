use crate::translate::build_model as bm;
use crate::translate::normalize::normalize_rules;
use crate::translate::pddl_ast::Condition;
use crate::translate::pddl_parser::SExpr;
use std::collections::{HashMap, HashSet};

// Minimal, compiling placeholder. We'll expand to mirror python/translate/pddl_to_prolog.py

fn sanitize(a: &str) -> String {
    a.replace('-', "_").to_lowercase()
}

// Parse typed lists: a b - t c - u  => [(a,Some(t)),(b,Some(t)),(c,Some(u))]
fn parse_typed_list(list: &[SExpr]) -> Vec<(String, Option<String>)> {
    let mut out = Vec::new();
    let mut buf: Vec<String> = Vec::new();
    let mut i = 0usize;
    while i < list.len() {
        match &list[i] {
            SExpr::Atom(a) if a == "-" => {
                if i + 1 < list.len() {
                    if let SExpr::Atom(t) = &list[i + 1] {
                        for name in buf.drain(..) {
                            out.push((name, Some(t.clone())));
                        }
                        i += 2;
                        continue;
                    }
                }
                i += 1;
            }
            SExpr::Atom(a) => {
                buf.push(a.clone());
                i += 1;
            }
            _ => {
                i += 1;
            }
        }
    }
    for name in buf.drain(..) {
        out.push((name, None));
    }
    out
}

fn format_sexpr_term(s: &SExpr) -> String {
    match s {
        SExpr::Atom(a) => sanitize(a),
        SExpr::List(list) => {
            if list.is_empty() {
                return "list()".to_string();
            }
            if let SExpr::Atom(head) = &list[0] {
                if head == "=" && list.len() == 3 {
                    if let SExpr::List(lhs) = &list[1] {
                        if let SExpr::Atom(fname) = &lhs[0] {
                            let args: Vec<String> =
                                lhs[1..].iter().map(|x| format_sexpr_term(x)).collect();
                            let rhs = format_sexpr_term(&list[2]);
                            return format!(
                                "assign({},{},{})",
                                sanitize(fname),
                                args.join(","),
                                rhs
                            );
                        }
                    }
                }
                let args: Vec<String> = list[1..].iter().map(|x| format_sexpr_term(x)).collect();
                format!("{}({})", sanitize(head), args.join(", "))
            } else {
                let parts: Vec<String> = list.iter().map(|x| format_sexpr_term(x)).collect();
                format!("list({})", parts.join(", "))
            }
        }
    }
}

pub fn domain_to_prolog(forms: &[SExpr]) -> String {
    let mut lines: Vec<String> = Vec::new();
    for f in forms {
        if let SExpr::List(items) = f {
            if items.len() >= 2 {
                if let Some(SExpr::Atom(a0)) = items.get(0) {
                    if a0.eq_ignore_ascii_case("define") {
                        if let Some(SExpr::List(domain_spec)) = items.get(1) {
                            if domain_spec.len() >= 2 {
                                if let SExpr::Atom(domain_kw) = &domain_spec[0] {
                                    if domain_kw.eq_ignore_ascii_case("domain") {
                                        if let SExpr::Atom(name) = &domain_spec[1] {
                                            lines.push(format!("domain({}).", sanitize(name)));
                                        }
                                    }
                                }
                            }
                        }
                        for section in &items[2..] {
                            if let SExpr::List(list) = section {
                                if list.is_empty() {
                                    continue;
                                }
                                if let SExpr::Atom(key) = &list[0] {
                                    match key.to_lowercase().as_str() {
                                        ":predicates" | "predicates" => {
                                            for item in &list[1..] {
                                                if let SExpr::List(p) = item {
                                                    if !p.is_empty() {
                                                        if let SExpr::Atom(nm) = &p[0] {
                                                            let params = parse_typed_list(&p[1..]);
                                                            lines.push(format!(
                                                                "predicate({},{}).",
                                                                sanitize(nm),
                                                                params.len()
                                                            ));
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                        ":functions" | "functions" => {
                                            for item in &list[1..] {
                                                match item {
                                                    SExpr::List(p) => {
                                                        if !p.is_empty() {
                                                            if let SExpr::Atom(nm) = &p[0] {
                                                                let params =
                                                                    parse_typed_list(&p[1..]);
                                                                lines.push(format!(
                                                                    "function({},{}).",
                                                                    sanitize(nm),
                                                                    params.len()
                                                                ));
                                                            }
                                                        }
                                                    }
                                                    SExpr::Atom(nm) => lines.push(format!(
                                                        "function({},0).",
                                                        sanitize(nm)
                                                    )),
                                                }
                                            }
                                        }
                                        ":action" | "action" => {
                                            if list.len() >= 2 {
                                                if let SExpr::Atom(name) = &list[1] {
                                                    let aname = sanitize(name);
                                                    lines.push(format!("action({}).", aname));
                                                    let mut i = 2usize;
                                                    while i < list.len() {
                                                        if let SExpr::Atom(k) = &list[i] {
                                                            let kl = k.to_lowercase();
                                                            if kl == ":parameters"
                                                                && i + 1 < list.len()
                                                            {
                                                                if let SExpr::List(params) =
                                                                    &list[i + 1]
                                                                {
                                                                    let parsed =
                                                                        parse_typed_list(params);
                                                                    for (idx, (pname, ptype)) in
                                                                        parsed.iter().enumerate()
                                                                    {
                                                                        match ptype {
                                                                            Some(t) => lines.push(format!("action_param({}, {}, {}, type({})).", aname, idx, pname, sanitize(t))),
                                                                            None => lines.push(format!("action_param({}, {}, {}).", aname, idx, pname)),
                                                                        }
                                                                    }
                                                                }
                                                                i += 2;
                                                                continue;
                                                            } else if kl == ":precondition"
                                                                && i + 1 < list.len()
                                                            {
                                                                if let Some(pre) = list.get(i + 1) {
                                                                    match pre {
                                                                        SExpr::List(pl) => {
                                                                            // assume (and ...)
                                                                            let mut it = pl.iter();
                                                                            let head = it.next();
                                                                            if let Some(
                                                                                SExpr::Atom(h),
                                                                            ) = head
                                                                            {
                                                                                if h.eq_ignore_ascii_case("and") {
                                                                                for item in it {
                                                                                    if let SExpr::List(a) = item {
                                                                                        if let Some(SExpr::Atom(pred)) = a.get(0) {
                                                                                            let args: Vec<String> = a[1..].iter().filter_map(|x| match x { SExpr::Atom(s) => Some(s.clone()), _ => None }).collect();
                                                                                            lines.push(format!("action_pre({}, {}({})).", aname, sanitize(pred), args.join(", ")));
                                                                                        }
                                                                                    }
                                                                                }
                                                                            } else {
                                                                                // single atom precondition: (pred args...)
                                                                                if let Some(SExpr::Atom(pred)) = pl.get(0) {
                                                                                    let args: Vec<String> = pl[1..].iter().filter_map(|x| match x { SExpr::Atom(s)=>Some(s.clone()), _=>None }).collect();
                                                                                    lines.push(format!("action_pre({}, {}({})).", aname, sanitize(pred), args.join(", ")));
                                                                                }
                                                                            }
                                                                            }
                                                                        }
                                                                        _ => {}
                                                                    }
                                                                }
                                                                i += 2;
                                                                continue;
                                                            } else if kl == ":effect"
                                                                && i + 1 < list.len()
                                                            {
                                                                if let Some(eff) = list.get(i + 1) {
                                                                    if let SExpr::List(el) = eff {
                                                                        let mut it = el.iter();
                                                                        let head = it.next();
                                                                        if let Some(SExpr::Atom(
                                                                            h,
                                                                        )) = head
                                                                        {
                                                                            if h.eq_ignore_ascii_case("and") {
                                                                            for item in it {
                                                                                match item {
                                                                                    SExpr::List(a) => {
                                                                                        if let Some(SExpr::Atom(tag)) = a.get(0) {
                                                                                            match tag.to_lowercase().as_str() {
                                                                                                "not" => {
                                                                                                    if let Some(SExpr::List(inner)) = a.get(1) {
                                                                                                        if let Some(SExpr::Atom(pred)) = inner.get(0) {
                                                                                                            let args: Vec<String> = inner[1..].iter().filter_map(|x| match x { SExpr::Atom(s)=>Some(s.clone()), _=>None }).collect();
                                                                                                            lines.push(format!("action_eff_del({}, {}({})).", aname, sanitize(pred), args.join(", ")));
                                                                                                        }
                                                                                                    }
                                                                                                }
                                                                                                "increase" | "decrease" => {
                                                                                                    if a.len() >= 3 {
                                                                                                        if let SExpr::List(flst) = &a[1] {
                                                                                                            if let Some(SExpr::Atom(fname)) = flst.get(0) {
                                                                                                                if let SExpr::Atom(v) = &a[2] {
                                                                                                                    lines.push(format!("action_eff_num({}, {}, {}).", aname, sanitize(fname), v));
                                                                                                                }
                                                                                                            }
                                                                                                        }
                                                                                                    }
                                                                                                }
                                                                                                _ => {
                                                                                                    let pred = tag;
                                                                                                    let args: Vec<String> = a[1..].iter().filter_map(|x| match x { SExpr::Atom(s)=>Some(s.clone()), _=>None }).collect();
                                                                                                    lines.push(format!("action_eff_add({}, {}({})).", aname, sanitize(pred), args.join(", ")));
                                                                                                }
                                                                                            }
                                                                                        }
                                                                                    }
                                                                                    SExpr::Atom(a) => {
                                                                                        lines.push(format!("action_eff_add({}, {}).", aname, sanitize(a)));
                                                                                    }
                                                                                }
                                                                            }
                                                                        } else {
                                                                            // single effect
                                                                            if let Some(SExpr::Atom(pred)) = head {
                                                                                let args: Vec<String> = el[1..].iter().filter_map(|x| match x { SExpr::Atom(s)=>Some(s.clone()), _=>None }).collect();
                                                                                lines.push(format!("action_eff_add({}, {}({})).", aname, sanitize(pred), args.join(", ")));
                                                                            }
                                                                        }
                                                                        }
                                                                    }
                                                                }
                                                                i += 2;
                                                                continue;
                                                            }
                                                        }
                                                        i += 1;
                                                    }
                                                }
                                            }
                                        }
                                        _ => {}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    lines.sort();
    lines.dedup();
    lines.join("\n")
}

pub fn problem_to_prolog(forms: &[SExpr]) -> String {
    let mut lines: Vec<String> = Vec::new();
    for f in forms {
        if let SExpr::List(items) = f {
            if items.len() >= 2 {
                if let Some(SExpr::Atom(a0)) = items.get(0) {
                    if a0.eq_ignore_ascii_case("define") {
                        for part in &items[1..] {
                            if let SExpr::List(inner) = part {
                                if inner.is_empty() {
                                    continue;
                                }
                                if let SExpr::Atom(atom0) = &inner[0] {
                                    match atom0.to_lowercase().as_str() {
                                        ":problem" | "problem" => {
                                            if inner.len() >= 2 {
                                                if let SExpr::Atom(n) = &inner[1] {
                                                    lines
                                                        .push(format!("problem({}).", sanitize(n)));
                                                }
                                            }
                                        }
                                        ":objects" | "objects" => {
                                            let typed = parse_typed_list(&inner[1..]);
                                            for (n, t) in typed {
                                                if let Some(tp) = t {
                                                    lines.push(format!(
                                                        "object({}, type({})).",
                                                        sanitize(&n),
                                                        sanitize(&tp)
                                                    ));
                                                } else {
                                                    lines
                                                        .push(format!("object({}).", sanitize(&n)));
                                                }
                                            }
                                        }
                                        ":init" | "init" => {
                                            for token in &inner[1..] {
                                                match token {
                                                    SExpr::List(lst) => {
                                                        if let Some(SExpr::Atom(name)) = lst.get(0)
                                                        {
                                                            let args: Vec<String> = lst[1..]
                                                                .iter()
                                                                .map(|a| format_sexpr_term(a))
                                                                .collect();
                                                            lines.push(format!(
                                                                "init({}({})).",
                                                                sanitize(name),
                                                                args.join(", ")
                                                            ));
                                                        }
                                                    }
                                                    SExpr::Atom(a) => lines
                                                        .push(format!("init({}).", sanitize(a))),
                                                }
                                            }
                                        }
                                        ":goal" | "goal" => {
                                            if inner.len() >= 2 {
                                                match &inner[1] {
                                                    SExpr::List(g) => {
                                                        let mut it = g.iter();
                                                        let head = it.next();
                                                        if let Some(SExpr::Atom(h)) = head {
                                                            if h.eq_ignore_ascii_case("and") {
                                                                for item in it {
                                                                    if let SExpr::List(atom) = item
                                                                    {
                                                                        if let Some(SExpr::Atom(
                                                                            gn,
                                                                        )) = atom.get(0)
                                                                        {
                                                                            let args: Vec<String> = atom[1..].iter().map(|a| format_sexpr_term(a)).collect();
                                                                            lines.push(format!(
                                                                                "goal({}({})).",
                                                                                sanitize(gn),
                                                                                args.join(", ")
                                                                            ));
                                                                        }
                                                                    }
                                                                }
                                                            } else {
                                                                if let Some(SExpr::Atom(gn)) = head
                                                                {
                                                                    let args: Vec<String> = g[1..]
                                                                        .iter()
                                                                        .map(|a| {
                                                                            format_sexpr_term(a)
                                                                        })
                                                                        .collect();
                                                                    lines.push(format!(
                                                                        "goal({}({})).",
                                                                        sanitize(gn),
                                                                        args.join(", ")
                                                                    ));
                                                                }
                                                            }
                                                        }
                                                    }
                                                    SExpr::Atom(a) => lines
                                                        .push(format!("goal({}).", sanitize(a))),
                                                }
                                            }
                                        }
                                        ":metric" | "metric" => {
                                            if inner.len() >= 2 {
                                                let metric_kind = match &inner[1] {
                                                    SExpr::Atom(k) => sanitize(k),
                                                    _ => "minimize".to_string(),
                                                };
                                                let metric_expr = if inner.len() >= 3 {
                                                    format!("{}", format_sexpr_term(&inner[2]))
                                                } else {
                                                    String::new()
                                                };
                                                lines.push(format!(
                                                    "metric({}, {}).",
                                                    metric_kind, metric_expr
                                                ));
                                            }
                                        }
                                        _ => {}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    lines.sort();
    lines.dedup();
    lines.join("\n")
}

// ===== Python-like scaffolding based on python/translate/pddl_to_prolog.py =====

#[derive(Debug, Default)]
pub struct PrologProgram {
    pub facts: Vec<String>,
    pub rules: Vec<String>,
    pub objects: HashSet<String>,
    fact_set: HashSet<String>,
    model_fact_set: HashSet<bm::Atom>,
    counter: usize,
    // Structured representation for Rust model building
    pub model_facts: Vec<bm::Atom>,
    pub model_rules: Vec<bm::RuleSpec>,
}

impl PrologProgram {
    pub fn new() -> Self {
        Self {
            facts: vec![],
            rules: vec![],
            objects: HashSet::new(),
            fact_set: HashSet::new(),
            model_fact_set: HashSet::new(),
            counter: 0,
            model_facts: vec![],
            model_rules: vec![],
        }
    }
    pub fn new_name(&mut self) -> String {
        let n = self.counter;
        self.counter += 1;
        format!("p${}", n)
    }
    pub fn add_fact<S: Into<String>>(&mut self, atom: S) {
        let a = atom.into();
        if self.fact_set.insert(a.clone()) {
            self.facts.push(a);
        }
    }
    pub fn add_rule<S: Into<String>>(&mut self, rule: S) {
        self.rules.push(rule.into());
    }
    pub fn add_model_fact(&mut self, atom: bm::Atom) {
        if self.model_fact_set.insert(atom.clone()) {
            self.model_facts.push(atom);
        }
    }
    pub fn add_model_rule(&mut self, rule: bm::RuleSpec) {
        self.model_rules.push(rule);
    }
    pub fn dump(&self) -> String {
        let mut out = String::new();
        out.push_str("Facts in PrologProgram:\n");
        for f in &self.facts {
            out.push_str(f);
            out.push_str("\n");
        }
        out.push_str("Rules in PrologProgram:\n");
        for r in &self.rules {
            out.push_str(r);
            out.push_str("\n");
        }
        out
    }
    pub fn normalize(&mut self) {
        let outcome = normalize_rules(&mut self.model_rules);
        for fact in outcome.new_facts {
            if let Some(fact_str) = model_atom_to_fact(&fact) {
                self.add_fact(fact_str);
            }
            self.add_model_fact(fact);
        }
        if outcome.object_predicate_required {
            let snapshot: Vec<String> = self.objects.iter().cloned().collect();
            for obj in snapshot {
                let atom = bm::Atom {
                    predicate: "@object".to_string(),
                    args: vec![bm::Arg::Const(obj.clone())],
                };
                if let Some(fact_str) = model_atom_to_fact(&atom) {
                    self.add_fact(fact_str);
                }
                self.add_model_fact(atom);
            }
        }
    }
}

fn atom_to_string(pred: &str, args: &[String]) -> String {
    if args.is_empty() {
        format!("{}.", sanitize(pred))
    } else {
        format!(
            "{}({}).",
            sanitize(pred),
            args.iter()
                .map(|a| sanitize(a))
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}

fn translate_typed_object(
    prog: &mut PrologProgram,
    obj: &(String, Option<String>),
    _type_hierarchy: &HashMap<String, Vec<String>>,
) {
    let (name, ty) = obj;
    prog.objects.insert(name.clone());
    if let Some(t) = ty {
        // add fact type(name).
        prog.add_fact(atom_to_string(t, &[name.clone()]));
        // add supertypes if we had them; we don't yet compute hierarchy, so skip
        // structured model fact for datalog
        prog.add_model_fact(bm::Atom {
            predicate: t.clone(),
            args: vec![bm::Arg::Const(name.clone())],
        });
    }
}

fn sexpr_atom_to_fact(sexpr: &SExpr) -> Option<String> {
    match sexpr {
        SExpr::Atom(a) => Some(atom_to_string(a, &[])),
        SExpr::List(list) => {
            if let Some(SExpr::Atom(pred)) = list.get(0) {
                let mut args: Vec<String> = Vec::new();
                for a in &list[1..] {
                    if let SExpr::Atom(s) = a {
                        args.push(s.clone());
                    }
                }
                Some(atom_to_string(pred, &args))
            } else {
                None
            }
        }
    }
}

fn sexpr_atom_to_model_atom(sexpr: &SExpr) -> Option<bm::Atom> {
    match sexpr {
        SExpr::Atom(a) => Some(bm::Atom {
            predicate: a.clone(),
            args: vec![],
        }),
        SExpr::List(list) => {
            if let Some(SExpr::Atom(pred)) = list.get(0) {
                let mut args: Vec<bm::Arg> = Vec::new();
                for a in &list[1..] {
                    if let SExpr::Atom(s) = a {
                        args.push(bm::Arg::Const(s.clone()));
                    }
                }
                Some(bm::Atom {
                    predicate: pred.clone(),
                    args,
                })
            } else {
                None
            }
        }
    }
}

fn model_atom_to_fact(atom: &bm::Atom) -> Option<String> {
    let mut args: Vec<String> = Vec::new();
    for arg in &atom.args {
        match arg {
            bm::Arg::Const(val) => args.push(val.clone()),
            _ => return None,
        }
    }
    Some(atom_to_string(&atom.predicate, &args))
}

fn cond_to_symatoms(cond: &Condition) -> Option<Vec<bm::SymAtom>> {
    match cond {
        Condition::Atom(name, args) => Some(vec![bm::SymAtom::new(name.clone(), args.clone())]),
        Condition::And(list) => {
            let mut out: Vec<bm::SymAtom> = Vec::new();
            for c in list {
                if let Some(mut v) = cond_to_symatoms(c) {
                    out.append(&mut v);
                } else {
                    return None;
                }
            }
            Some(out)
        }
        Condition::True => Some(vec![]),
        _ => None, // skip Not/Comparison for now
    }
}

pub fn translate_from_ast(domain_forms: &[SExpr], problem_forms: &[SExpr]) -> PrologProgram {
    let mut prog = PrologProgram::new();
    let dom = crate::translate::pddl_ast::Domain::from_sexprs(domain_forms);
    let prob = crate::translate::pddl_ast::Problem::from_sexprs(problem_forms);
    if let (Some(dom), Some(prob)) = (dom, prob) {
        // Build a trivial type hierarchy map: type -> empty supertypes (full hierarchy not yet ported)
        let type_h: HashMap<String, Vec<String>> = HashMap::new();
        // objects
        for obj in &prob.objects {
            translate_typed_object(&mut prog, obj, &type_h);
        }
        // init facts
        for init in &prob.init {
            if let Some(f) = sexpr_atom_to_fact(init) {
                prog.add_fact(f);
            }
            if let Some(a) = sexpr_atom_to_model_atom(init) {
                prog.add_model_fact(a);
            }
        }
        // Numeric init isn't modeled separately in current AST; future work.
        // Build minimal exploration rules for actions with conjunction-of-atom preconditions
        for action in &dom.actions {
            if let Some(pre) = &action.precond {
                let cond = crate::translate::pddl_ast::sexpr_to_condition(pre);
                if let Some(conds) = cond_to_symatoms(&cond) {
                    // Head predicate name: action name. Ensure params are proper var tokens (don't double-prefix '?').
                    let head_args: Vec<String> = action
                        .parameters
                        .iter()
                        .map(|(p, _)| {
                            if p.starts_with('?') {
                                p.clone()
                            } else {
                                format!("?{}", p)
                            }
                        })
                        .collect();
                    let head = bm::SymAtom::new(action.name.clone(), head_args);
                    let rtype = match conds.len() {
                        0 => "project", // trivial; will turn into facts after convert_trivial_rules in Python; we keep project
                        1 => "project",
                        2 => "join",
                        _ => "product", // approximation; proper splitting not yet ported
                    };
                    prog.add_model_rule(bm::RuleSpec {
                        rtype: rtype.to_string(),
                        effect: head,
                        conditions: conds,
                    });
                }
            }
        }
    }
    prog.normalize();
    prog
}

// Convenience: compute a datalog model directly from AST using our minimal rules
pub fn compute_model_from_ast(
    domain_forms: &[crate::translate::pddl_parser::SExpr],
    problem_forms: &[crate::translate::pddl_parser::SExpr],
) -> Vec<bm::Atom> {
    let prog = translate_from_ast(domain_forms, problem_forms);
    // Convert rules and facts to engine input
    let mut rules = bm::convert_rules(&prog.model_rules);
    bm::compute_model(&mut rules, &prog.model_facts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::translate::pddl_parser::SExpr;

    fn a(s: &str) -> SExpr {
        SExpr::Atom(s.to_string())
    }
    fn l(items: Vec<SExpr>) -> SExpr {
        SExpr::List(items)
    }

    #[test]
    fn compute_model_with_single_precondition_action() {
        // Domain with an action move(?x) pre: at(?x)
        let domain = vec![l(vec![
            a("define"),
            l(vec![a("domain"), a("d")]),
            // action inside define (flat tokens, as expected by parser)
            l(vec![
                a(":action"),
                a("move"),
                a(":parameters"),
                l(vec![a("?x")]),
                a(":precondition"),
                l(vec![a("at"), a("?x")]),
            ]),
        ])];
        let problem = vec![l(vec![
            a("define"),
            l(vec![a("problem"), a("p")]),
            l(vec![a(":init"), l(vec![a("at"), a("a1")])]),
        ])];
        let model = compute_model_from_ast(&domain, &problem);
        // Expect at(a1) fact present and move(a1) derived
        let has_at = model
            .iter()
            .any(|m| m.predicate == "at" && matches!(&m.args[0], bm::Arg::Const(s) if s=="a1"));
        assert!(has_at);
        let has_move = model
            .iter()
            .any(|m| m.predicate == "move" && matches!(&m.args[0], bm::Arg::Const(s) if s=="a1"));
        assert!(has_move);
    }

    #[test]
    fn normalize_adds_object_guard_and_fact() {
        let mut prog = PrologProgram::new();
        prog.objects.insert("a1".to_string());
        prog.add_model_rule(bm::RuleSpec {
            rtype: "project".to_string(),
            effect: bm::SymAtom::new("move", vec!["?x"]),
            conditions: vec![],
        });

        prog.normalize();

        assert_eq!(prog.model_rules.len(), 1);
        let conds = &prog.model_rules[0].conditions;
        assert!(conds
            .iter()
            .any(|c| c.predicate == "@object" && c.args == vec!["?x".to_string()]));
        let has_object_fact = prog.model_facts.iter().any(|a| {
            a.predicate == "@object"
                && matches!(a.args.get(0), Some(bm::Arg::Const(val)) if val == "a1")
        });
        assert!(has_object_fact);
        assert!(prog.facts.iter().any(|f| f == "@object(a1)."));
    }

    #[test]
    fn normalize_converts_trivial_rule_to_fact() {
        let mut prog = PrologProgram::new();
        prog.add_model_rule(bm::RuleSpec {
            rtype: "project".to_string(),
            effect: bm::SymAtom::new("ready", vec![]),
            conditions: vec![],
        });

        prog.normalize();

        assert!(prog.model_rules.is_empty());
        assert!(prog.model_facts.iter().any(|a| a.predicate == "ready"));
        assert!(prog.facts.iter().any(|f| f == "ready."));
    }
}
