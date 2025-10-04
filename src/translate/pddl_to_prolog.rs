use crate::translate::pddl_parser::SExpr;
use std::collections::{HashMap, HashSet};
use crate::translate::pddl_ast::{Domain, Problem};

// Minimal, compiling placeholder. We'll expand to mirror python/translate/pddl_to_prolog.py

fn sanitize(a: &str) -> String { a.replace('-', "_").to_lowercase() }

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
                        for name in buf.drain(..) { out.push((name, Some(t.clone()))); }
                        i += 2;
                        continue;
                    }
                }
                i += 1;
            }
            SExpr::Atom(a) => { buf.push(a.clone()); i += 1; }
            _ => { i += 1; }
        }
    }
    for name in buf.drain(..) { out.push((name, None)); }
    out
}

fn format_sexpr_term(s: &SExpr) -> String {
    match s {
        SExpr::Atom(a) => sanitize(a),
        SExpr::List(list) => {
            if list.is_empty() { return "list()".to_string(); }
            if let SExpr::Atom(head) = &list[0] {
                if head == "=" && list.len() == 3 {
                    if let SExpr::List(lhs) = &list[1] {
                        if let SExpr::Atom(fname) = &lhs[0] {
                            let args: Vec<String> = lhs[1..].iter().map(|x| format_sexpr_term(x)).collect();
                            let rhs = format_sexpr_term(&list[2]);
                            return format!("assign({},{},{})", sanitize(fname), args.join(","), rhs);
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
                                if list.is_empty() { continue; }
                                if let SExpr::Atom(key) = &list[0] {
                                    match key.to_lowercase().as_str() {
                                        ":predicates" | "predicates" => {
                                            for item in &list[1..] {
                                                if let SExpr::List(p) = item { if !p.is_empty() {
                                                    if let SExpr::Atom(nm) = &p[0] {
                                                        let params = parse_typed_list(&p[1..]);
                                                        lines.push(format!("predicate({},{}).", sanitize(nm), params.len()));
                                                    }
                                                }}
                                            }
                                        }
                                        ":functions" | "functions" => {
                                            for item in &list[1..] {
                                                match item {
                                                    SExpr::List(p) => { if !p.is_empty() {
                                                        if let SExpr::Atom(nm) = &p[0] {
                                                            let params = parse_typed_list(&p[1..]);
                                                            lines.push(format!("function({},{}).", sanitize(nm), params.len()));
                                                        }
                                                    }}
                                                    SExpr::Atom(nm) => lines.push(format!("function({},0).", sanitize(nm))),
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
                                                            if kl == ":parameters" && i + 1 < list.len() {
                                                                if let SExpr::List(params) = &list[i + 1] {
                                                                    let parsed = parse_typed_list(params);
                                                                    for (idx, (pname, ptype)) in parsed.iter().enumerate() {
                                                                        match ptype {
                                                                            Some(t) => lines.push(format!("action_param({}, {}, {}, type({})).", aname, idx, pname, sanitize(t))),
                                                                            None => lines.push(format!("action_param({}, {}, {}).", aname, idx, pname)),
                                                                        }
                                                                    }
                                                                }
                                                                i += 2; continue;
                                                            } else if kl == ":precondition" && i + 1 < list.len() {
                                                                if let Some(pre) = list.get(i + 1) {
                                                                    match pre {
                                                                        SExpr::List(pl) => {
                                                                            // assume (and ...)
                                                                            let mut it = pl.iter();
                                                                            let head = it.next();
                                                                            if let Some(SExpr::Atom(h)) = head { if h.eq_ignore_ascii_case("and") {
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
                                                                            }}
                                                                        }
                                                                        _ => {}
                                                                    }
                                                                }
                                                                i += 2; continue;
                                                            } else if kl == ":effect" && i + 1 < list.len() {
                                                                if let Some(eff) = list.get(i + 1) {
                                                                    if let SExpr::List(el) = eff {
                                                                        let mut it = el.iter();
                                                                        let head = it.next();
                                                                        if let Some(SExpr::Atom(h)) = head { if h.eq_ignore_ascii_case("and") {
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
                                                                        }}
                                                                    }
                                                                }
                                                                i += 2; continue;
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
                                if inner.is_empty() { continue; }
                                if let SExpr::Atom(atom0) = &inner[0] {
                                    match atom0.to_lowercase().as_str() {
                                        ":problem" | "problem" => {
                                            if inner.len() >= 2 {
                                                if let SExpr::Atom(n) = &inner[1] { lines.push(format!("problem({}).", sanitize(n))); }
                                            }
                                        }
                                        ":objects" | "objects" => {
                                            let typed = parse_typed_list(&inner[1..]);
                                            for (n, t) in typed {
                                                if let Some(tp) = t { lines.push(format!("object({}, type({})).", sanitize(&n), sanitize(&tp))); }
                                                else { lines.push(format!("object({}).", sanitize(&n))); }
                                            }
                                        }
                                        ":init" | "init" => {
                                            for token in &inner[1..] {
                                                match token {
                                                    SExpr::List(lst) => {
                                                        if let Some(SExpr::Atom(name)) = lst.get(0) {
                                                            let args: Vec<String> = lst[1..].iter().map(|a| format_sexpr_term(a)).collect();
                                                            lines.push(format!("init({}({})).", sanitize(name), args.join(", ")));
                                                        }
                                                    }
                                                    SExpr::Atom(a) => lines.push(format!("init({}).", sanitize(a))),
                                                }
                                            }
                                        }
                                        ":goal" | "goal" => {
                                            if inner.len() >= 2 {
                                                match &inner[1] {
                                                    SExpr::List(g) => {
                                                        let mut it = g.iter();
                                                        let head = it.next();
                                                        if let Some(SExpr::Atom(h)) = head { if h.eq_ignore_ascii_case("and") {
                                                            for item in it {
                                                                if let SExpr::List(atom) = item {
                                                                    if let Some(SExpr::Atom(gn)) = atom.get(0) {
                                                                        let args: Vec<String> = atom[1..].iter().map(|a| format_sexpr_term(a)).collect();
                                                                        lines.push(format!("goal({}({})).", sanitize(gn), args.join(", ")));
                                                                    }
                                                                }
                                                            }
                                                        } else {
                                                            if let Some(SExpr::Atom(gn)) = head { let args: Vec<String> = g[1..].iter().map(|a| format_sexpr_term(a)).collect(); lines.push(format!("goal({}({})).", sanitize(gn), args.join(", "))); }
                                                        }}
                                                    }
                                                    SExpr::Atom(a) => lines.push(format!("goal({}).", sanitize(a))),
                                                }
                                            }
                                        }
                                        ":metric" | "metric" => {
                                            if inner.len() >= 2 {
                                                let metric_kind = match &inner[1] { SExpr::Atom(k) => sanitize(k), _ => "minimize".to_string() };
                                                let metric_expr = if inner.len() >= 3 { format!("{}", format_sexpr_term(&inner[2])) } else { String::new() };
                                                lines.push(format!("metric({}, {}).", metric_kind, metric_expr));
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
    counter: usize,
}

impl PrologProgram {
    pub fn new() -> Self { Self { facts: vec![], rules: vec![], objects: HashSet::new(), counter: 0 } }
    pub fn new_name(&mut self) -> String { let n = self.counter; self.counter += 1; format!("p${}", n) }
    pub fn add_fact<S: Into<String>>(&mut self, atom: S) { let a = atom.into(); self.facts.push(a.clone()); }
    pub fn add_rule<S: Into<String>>(&mut self, rule: S) { self.rules.push(rule.into()); }
    pub fn dump(&self) -> String {
        let mut out = String::new();
        out.push_str("Facts in PrologProgram:\n");
        for f in &self.facts { out.push_str(f); out.push_str("\n"); }
        out.push_str("Rules in PrologProgram:\n");
        for r in &self.rules { out.push_str(r); out.push_str("\n"); }
        out
    }
    pub fn normalize(&mut self) {
        // Placeholder: in Python this removes free effect variables, splits duplicate args, converts trivial rules.
        // We don't build rules yet, so nothing to do here.
    }
}

fn atom_to_string(pred: &str, args: &[String]) -> String {
    if args.is_empty() { format!("{}.", sanitize(pred)) } else { format!("{}({}).", sanitize(pred), args.iter().map(|a| sanitize(a)).collect::<Vec<_>>().join(", ")) }
}

fn translate_typed_object(prog: &mut PrologProgram, obj: &(String, Option<String>), type_hierarchy: &HashMap<String, Vec<String>>) {
    let (name, ty) = obj;
    if let Some(t) = ty {
        // add fact type(name).
        prog.add_fact(atom_to_string(t, &[name.clone()]));
        // add supertypes if we had them; we don't yet compute hierarchy, so skip
    }
}

fn sexpr_atom_to_fact(sexpr: &SExpr) -> Option<String> {
    match sexpr {
        SExpr::Atom(a) => Some(atom_to_string(a, &[])),
        SExpr::List(list) => {
            if let Some(SExpr::Atom(pred)) = list.get(0) {
                let mut args: Vec<String> = Vec::new();
                for a in &list[1..] { if let SExpr::Atom(s) = a { args.push(s.clone()); } }
                Some(atom_to_string(pred, &args))
            } else { None }
        }
    }
}

pub fn translate_from_ast(domain_forms: &[SExpr], problem_forms: &[SExpr]) -> PrologProgram {
    let mut prog = PrologProgram::new();
    let dom = Domain::from_sexprs(domain_forms);
    let prob = Problem::from_sexprs(problem_forms);
    if let (Some(dom), Some(prob)) = (dom, prob) {
        // Build a trivial type hierarchy map: type -> empty supertypes (full hierarchy not yet ported)
        let mut type_h: HashMap<String, Vec<String>> = HashMap::new();
        // objects
        for obj in &prob.objects { translate_typed_object(&mut prog, obj, &type_h); }
        // init facts
        for init in &prob.init { if let Some(f) = sexpr_atom_to_fact(init) { prog.add_fact(f); } }
        // Numeric init isn't modeled separately in current AST; future work.
        // No rules yet; would require normalize/build_exploration_rules port.
        let _ = dom; // silence unused
    }
    prog.normalize();
    prog
}
