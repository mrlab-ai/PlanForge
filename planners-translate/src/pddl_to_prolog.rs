/// Port of pddl_to_prolog.py
/// Translates a PDDL task into a logic program for grounding.
use std::collections::{HashMap, HashSet};

use log::info;

use super::normalize;
use super::pddl::conditions::*;
use super::pddl::pddl_types::{TypedObject, get_type_predicate_name};
use super::pddl::tasks::Task;

/// Python: class Fact(object)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Fact {
    pub atom: Vec<String>, // predicate name followed by args
}

impl Fact {
    pub fn new(atom: Vec<String>) -> Self {
        Fact { atom }
    }
}

/// Rule type for build_model dispatch
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuleType {
    Join,
    Product,
    Project,
}

/// Python: class Rule(object)
#[derive(Debug, Clone)]
pub struct Rule {
    pub conditions: Vec<Vec<String>>, // each condition is [pred, arg1, arg2, ...]
    pub effect: Vec<String>,          // [pred, arg1, arg2, ...]
    pub rule_type: Option<RuleType>,  // set by split_rules / greedy_join
}

impl Rule {
    pub fn new(conditions: Vec<Vec<String>>, effect: Vec<String>) -> Self {
        Rule {
            conditions,
            effect,
            rule_type: None,
        }
    }

    pub fn new_typed(
        conditions: Vec<Vec<String>>,
        effect: Vec<String>,
        rule_type: RuleType,
    ) -> Self {
        Rule {
            conditions,
            effect,
            rule_type: Some(rule_type),
        }
    }

    fn rename_duplicate_variables_in_atom(
        atom: &mut Vec<String>,
        extra_conditions: &mut Vec<(String, String)>,
    ) {
        let mut used_variables: HashSet<String> = HashSet::new();
        for (index, arg) in atom[1..].iter_mut().enumerate() {
            if arg.starts_with('?') {
                if used_variables.contains(arg) {
                    let new_name = format!("{}@{}", arg, extra_conditions.len());
                    let original = arg.clone();
                    *arg = new_name.clone();
                    extra_conditions.push((original, new_name));
                } else {
                    used_variables.insert(arg.clone());
                }
            }
        }
    }

    /// Python: def rename_duplicate_variables(self)
    pub fn rename_duplicate_variables(&mut self) {
        let mut extra_conditions: Vec<(String, String)> = vec![];

        Self::rename_duplicate_variables_in_atom(&mut self.effect, &mut extra_conditions);
        for cond in &mut self.conditions {
            Self::rename_duplicate_variables_in_atom(cond, &mut extra_conditions);
        }

        for (original, renamed) in extra_conditions {
            self.conditions
                .push(vec!["=".to_string(), original, renamed]);
        }
    }
}

/// Python: def get_variables(symbolic_atoms)
/// Get all variables (strings starting with '?') from a list of atoms.
pub fn get_variables(atoms: &[Vec<String>]) -> HashSet<String> {
    let mut variables = HashSet::new();
    for atom in atoms {
        for arg in &atom[1..] {
            if arg.starts_with('?') {
                variables.insert(arg.clone());
            }
        }
    }
    variables
}

/// Python: class PrologProgram(object)
pub struct PrologProgram {
    pub facts: HashSet<Fact>,
    pub rules: Vec<Rule>,
    pub objects: HashSet<String>,
}

impl PrologProgram {
    pub fn new() -> Self {
        PrologProgram {
            facts: HashSet::new(),
            rules: vec![],
            objects: HashSet::new(),
        }
    }

    pub fn add_fact(&mut self, atom: Vec<String>) {
        self.facts.insert(Fact::new(atom));
    }

    pub fn add_rule(&mut self, rule: Rule) {
        self.rules.push(rule);
    }

    /// Python: def normalize(self)
    pub fn normalize(&mut self) {
        // 1. Remove free effect variables
        self.remove_free_effect_variables();
        // 2. Split duplicate arguments
        self.split_duplicate_arguments();
        // 3. Convert trivial rules (empty conditions) into facts
        self.convert_trivial_rules();
    }

    /// Python: def split_rules(self)
    pub fn split_rules(&mut self) {
        let mut new_rules = vec![];
        let mut counter = 0;
        for rule in &self.rules {
            let split = super::split_rules::split_rule(rule, &mut counter);
            new_rules.extend(split);
        }
        self.rules = new_rules;
    }

    /// Python: def remove_free_effect_variables(self)
    pub fn remove_free_effect_variables(&mut self) {
        let mut must_add_predicate = false;
        for rule in &mut self.rules {
            let eff_vars: HashSet<String> = rule.effect[1..]
                .iter()
                .filter(|a| a.starts_with('?'))
                .cloned()
                .collect();
            let cond_vars: HashSet<String> = rule
                .conditions
                .iter()
                .flat_map(|c| c[1..].iter().filter(|a| a.starts_with('?')).cloned())
                .collect();

            let free_vars: Vec<String> = eff_vars.difference(&cond_vars).cloned().collect();
            if !free_vars.is_empty() {
                must_add_predicate = true;
                for var in free_vars {
                    rule.conditions.push(vec!["@object".to_string(), var]);
                }
            }
        }
        if must_add_predicate {
            info!("Unbound effect variables: Adding @object predicate.");
            for obj in self.objects.clone() {
                self.add_fact(vec!["@object".to_string(), obj]);
            }
        }
    }

    /// Python: def split_duplicate_arguments(self)
    pub fn split_duplicate_arguments(&mut self) {
        for rule in &mut self.rules {
            rule.rename_duplicate_variables();
        }
    }

    /// Python: def convert_trivial_rules(self)
    pub fn convert_trivial_rules(&mut self) {
        // Convert rules with no conditions to facts
        let mut new_facts = vec![];
        let mut new_rules = vec![];
        for rule in &self.rules {
            if rule.conditions.is_empty() {
                // Only convert to fact if there are no variables in effect
                if rule.effect[1..].iter().all(|a| !a.starts_with('?')) {
                    new_facts.push(rule.effect.clone());
                } else {
                    new_rules.push(rule.clone());
                }
            } else {
                new_rules.push(rule.clone());
            }
        }
        self.rules = new_rules;
        for fact in new_facts {
            self.add_fact(fact);
        }
    }
}

/// Python: def translate_typed_object(prog, obj, type_dict)
fn translate_typed_object(
    obj: &TypedObject,
    type_dict: &HashMap<String, &super::pddl::pddl_types::Type>,
    program: &mut PrologProgram,
) {
    program.objects.insert(obj.name.clone());
    // Add type atom for the object's own type and all supertypes
    let mut type_name = Some(obj.type_name.clone());
    while let Some(ref tn) = type_name {
        let type_pred = get_type_predicate_name(tn);
        program.add_fact(vec![type_pred, obj.name.clone()]);
        type_name = type_dict.get(tn).and_then(|t| t.basetype_name.clone());
    }
}

/// Python: def translate_facts(task, program)
fn translate_facts(task: &Task, program: &mut PrologProgram) {
    for atom in &task.init {
        let mut fact = vec![atom.predicate.clone()];
        fact.extend(atom.args.clone());
        program.add_fact(fact);
    }
    for assign in &task.num_init {
        let mut fact = vec![normalize::get_function_predicate(&assign.fluent.symbol)];
        fact.extend(assign.fluent.args.clone());
        program.add_fact(fact);
    }
}

/// Python: def translate(task) -> PrologProgram
/// Main translation function: converts PDDL task to a logic program.
pub fn translate(task: &Task) -> PrologProgram {
    let mut program = PrologProgram::new();

    // Build type dictionary
    let type_dict: HashMap<String, &super::pddl::pddl_types::Type> =
        task.types.iter().map(|t| (t.name.clone(), t)).collect();

    // Add objects with type facts
    for obj in &task.objects {
        translate_typed_object(obj, &type_dict, &mut program);
    }

    // Add init facts
    translate_facts(task, &mut program);

    for rule in normalize::build_exploration_rules(task) {
        let conditions = rule
            .conditions
            .iter()
            .flat_map(condition_to_atoms)
            .collect();
        let effect_atoms = condition_to_atoms(&rule.effect);
        if let Some(effect) = effect_atoms.into_iter().next() {
            program.add_rule(Rule::new(conditions, effect));
        }
    }

    // Normalize the program (steps 1-3 from Python)
    program.normalize();
    // Split rules (step 4: split n-ary joins into binary)
    program.split_rules();

    program
}

/// Convert a condition tree to a list of atoms for rules.
fn condition_to_atoms(cond: &Condition) -> Vec<Vec<String>> {
    match cond {
        Condition::Truth => vec![],
        Condition::Conjunction(conj) => conj
            .parts
            .iter()
            .flat_map(|p| condition_to_atoms(p))
            .collect(),
        Condition::Atom(atom) => {
            let mut result = vec![atom.predicate.clone()];
            result.extend(atom.args.clone());
            vec![result]
        }
        Condition::NegatedAtom(natom) => {
            // Negated atoms are generally ignored in exploration (relaxation)
            vec![]
        }
        Condition::FunctionComparison(fc) => fc
            .parts
            .iter()
            .flat_map(|part| part.primitive_numeric_expressions())
            .map(|pne| {
                let mut result = vec![normalize::get_function_predicate(&pne.symbol)];
                result.extend(pne.args.clone());
                result
            })
            .collect(),
        Condition::NegatedFunctionComparison(nfc) => nfc
            .parts
            .iter()
            .flat_map(|part| part.primitive_numeric_expressions())
            .map(|pne| {
                let mut result = vec![normalize::get_function_predicate(&pne.symbol)];
                result.extend(pne.args.clone());
                result
            })
            .collect(),
        _ => vec![],
    }
}
