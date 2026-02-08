use std::collections::{HashMap, HashSet};

use crate::translate::build_model;
use crate::translate::normalize;
use crate::translate::pddl_parser::SExpr;
use crate::translate::split_rules::{
    add_object_conditions_to_rules, convert_trivial_rules_to_facts, split_duplicate_arguments,
    split_rule, RuleWithType, SymRule,
};
use crate::translate::timers;

#[derive(Debug, Clone)]
pub struct Fact {
    pub atom: build_model::Atom,
}

impl std::fmt::Display for Fact {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}." , format_atom(&self.atom))
    }
}

#[derive(Debug, Clone)]
pub struct Rule {
    pub conditions: Vec<build_model::SymAtom>,
    pub effect: build_model::SymAtom,
    pub rtype: String,
}

impl Rule {
    pub fn add_condition(&mut self, condition: build_model::SymAtom) {
        self.conditions.push(condition);
    }

    pub fn variables(&self) -> HashSet<String> {
        get_variables(self.conditions.iter().chain(std::iter::once(&self.effect)))
    }
}

impl std::fmt::Display for Rule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let conds = self
            .conditions
            .iter()
            .map(format_sym_atom)
            .collect::<Vec<_>>()
            .join(", ");
        write!(f, "{} :- {}.", format_sym_atom(&self.effect), conds)
    }
}

pub struct PrologProgram {
    pub facts: Vec<Fact>,
    pub rules: Vec<Rule>,
    pub objects: HashSet<String>,
    counter: usize,
}

impl PrologProgram {
    pub fn new() -> Self {
        Self {
            facts: Vec::new(),
            rules: Vec::new(),
            objects: HashSet::new(),
            counter: 0,
        }
    }

    pub fn add_fact(&mut self, atom: build_model::Atom) {
        for arg in &atom.args {
            if let build_model::Arg::Const(val) = arg {
                self.objects.insert(val.clone());
            }
        }
        self.facts.push(Fact { atom });
    }

    pub fn add_rule(&mut self, rule: Rule) {
        self.rules.push(rule);
    }

    pub fn normalize(&mut self) {
        let mut sym_rules: Vec<SymRule> = self
            .rules
            .iter()
            .cloned()
            .map(|r| SymRule {
                conditions: r.conditions,
                effect: r.effect,
            })
            .collect();

        for rule in sym_rules.iter_mut() {
            split_duplicate_arguments(rule);
        }

        let object_predicate_required = add_object_conditions_to_rules(&mut sym_rules);
        if object_predicate_required {
            for obj in self.objects.clone() {
                self.facts.push(Fact {
                    atom: build_model::Atom {
                        predicate: "@object".to_string(),
                        args: vec![build_model::Arg::Const(obj)],
                    },
                });
            }
        }

        let (sym_rules, extra_facts) = convert_trivial_rules_to_facts(&sym_rules);
        for atom in extra_facts {
            self.add_fact(atom);
        }

        let mut split_rules_out: Vec<RuleWithType> = Vec::new();
        for rule in &sym_rules {
            split_rules_out.extend(split_rule(rule, &mut self.counter));
        }

        self.rules = split_rules_out
            .into_iter()
            .map(|r| Rule {
                conditions: r.conditions,
                effect: r.effect,
                rtype: r.rtype,
            })
            .collect();
    }

    pub fn dump(&self) {
        println!("Facts in PrologProgram:");
        for fact in &self.facts {
            println!("{}", fact);
        }
        println!("Rules in PrologProgram:");
        for rule in &self.rules {
            println!("{}", rule);
        }
    }
}

fn format_atom(atom: &build_model::Atom) -> String {
    let args = atom
        .args
        .iter()
        .map(|arg| match arg {
            build_model::Arg::Const(val) => val.clone(),
            build_model::Arg::Var(idx) => format!("?{}", idx),
            build_model::Arg::FreeVar(name) => name.clone(),
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!("Atom {}({})", atom.predicate, args)
}

fn format_sym_atom(atom: &build_model::SymAtom) -> String {
    let args = atom.args.join(", ");
    format!("Atom {}({})", atom.predicate, args)
}

fn get_variables<'a, I>(atoms: I) -> HashSet<String>
where
    I: IntoIterator<Item = &'a build_model::SymAtom>,
{
    let mut variables = HashSet::new();
    for sym_atom in atoms {
        for arg in &sym_atom.args {
            if arg.starts_with('?') {
                variables.insert(arg.clone());
            }
        }
    }
    variables
}

fn build_type_hierarchy(
    types: &[(String, Option<String>)],
) -> HashMap<String, Vec<String>> {
    let mut parent_map: HashMap<String, String> = HashMap::new();
    for (t, parent) in types {
        if let Some(p) = parent {
            parent_map.insert(t.clone(), p.clone());
        }
    }

    let mut hierarchy: HashMap<String, Vec<String>> = HashMap::new();
    for (t, _) in types {
        let mut chain = Vec::new();
        let mut current = t.clone();
        while let Some(parent) = parent_map.get(&current) {
            chain.push(parent.clone());
            if parent == "object" {
                break;
            }
            current = parent.clone();
        }
        hierarchy.insert(t.clone(), chain);
    }
    hierarchy
}

fn add_init_facts(prog: &mut PrologProgram, task: &normalize::NormalizableTask) {
    let type_hierarchy = build_type_hierarchy(&task.types);

    for (obj_name, obj_type) in &task.objects {
        let type_name = obj_type.clone().unwrap_or_else(|| "object".to_string());
        prog.add_fact(build_model::Atom {
            predicate: type_name.clone(),
            args: vec![build_model::Arg::Const(obj_name.clone())],
        });
        prog.add_fact(build_model::Atom {
            predicate: "=".to_string(),
            args: vec![
                build_model::Arg::Const(obj_name.clone()),
                build_model::Arg::Const(obj_name.clone()),
            ],
        });
        if let Some(supertypes) = type_hierarchy.get(&type_name) {
            for supertype in supertypes {
                prog.add_fact(build_model::Atom {
                    predicate: supertype.clone(),
                    args: vec![build_model::Arg::Const(obj_name.clone())],
                });
            }
        } else if type_name != "object" {
            prog.add_fact(build_model::Atom {
                predicate: "object".to_string(),
                args: vec![build_model::Arg::Const(obj_name.clone())],
            });
        }
    }

    for init_sexpr in &task.init {
        if let SExpr::List(items) = init_sexpr {
            if items.len() >= 3 {
                if let SExpr::Atom(op) = &items[0] {
                    if op == "=" {
                        if let SExpr::List(func_items) = &items[1] {
                            if let Some(SExpr::Atom(fname)) = func_items.get(0) {
                                let func_args: Vec<build_model::Arg> = func_items[1..]
                                    .iter()
                                    .filter_map(|item| match item {
                                        SExpr::Atom(s) => Some(build_model::Arg::Const(s.clone())),
                                        _ => None,
                                    })
                                    .collect();
                                let defined_pred = format!("defined!{}", fname);
                                prog.add_fact(build_model::Atom {
                                    predicate: defined_pred,
                                    args: func_args,
                                });
                                continue;
                            }
                        }
                    }
                }
            }
        }

        if let Some(atom) = sexpr_to_atom(init_sexpr) {
            if atom.predicate != "=" {
                prog.add_fact(atom);
            }
        }
    }
}

fn sexpr_to_atom(sexpr: &SExpr) -> Option<build_model::Atom> {
    match sexpr {
        SExpr::List(items) if !items.is_empty() => {
            if let SExpr::Atom(pred) = &items[0] {
                let args = items[1..]
                    .iter()
                    .filter_map(|s| match s {
                        SExpr::Atom(a) => Some(build_model::Arg::Const(a.clone())),
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                Some(build_model::Atom {
                    predicate: pred.clone(),
                    args,
                })
            } else {
                None
            }
        }
        _ => None,
    }
}

pub fn translate(task: &normalize::NormalizableTask) -> PrologProgram {
    let mut prog = PrologProgram::new();

    timers::timing("Generating Datalog program", false, || {
        add_init_facts(&mut prog, task);
        for (conditions, effect) in normalize::build_exploration_rules(task).unwrap_or_default() {
            prog.add_rule(Rule {
                conditions,
                effect,
                rtype: String::new(),
            });
        }
    });

    timers::timing("Normalizing Datalog program", true, || {
        prog.normalize();
    });

    prog
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::translate::pddl::{Domain, Problem};

    #[test]
    fn translate_smoke() {
        let task = crate::translate::pddl_parser::PddlTask::from_files(
            std::path::Path::new("misc/plant-watering/domain.pddl"),
            std::path::Path::new("misc/plant-watering/prob_4_1_1.pddl"),
        )
        .unwrap();
        let dom = Domain::from_sexprs(&task.domain_forms).expect("domain parsed");
        let prob = Problem::from_sexprs(&task.problem_forms).expect("problem parsed");
        let mut norm_task = normalize::NormalizableTask::from_ast(&dom, &prob);
        normalize::normalize(&mut norm_task).expect("normalization failed");
        let prog = translate(&norm_task);
        assert!(!prog.facts.is_empty());
    }
}
