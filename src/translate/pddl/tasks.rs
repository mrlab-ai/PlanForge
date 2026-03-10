/// Port of pddl/tasks.py
use std::collections::{HashMap, HashSet};
use std::fmt;

use super::pddl_types::{Type, TypedObject};
use super::predicates::Predicate;
use super::functions::Function;
use super::conditions::{Condition, Conjunction, Atom, NegatedAtom};
use super::f_expression::{
    FunctionalExpression, PrimitiveNumericExpression, FunctionAssignment, NumericConstant,
    ArithmeticExpression,
};
use super::effects::Effect;
use super::actions::Action;
use super::axioms::{Axiom, NumericAxiom};

fn prettyprint(symbol: &str) -> String {
    match symbol {
        "-" => "difference".to_string(),
        "+" => "sum".to_string(),
        "*" => "product".to_string(),
        "/" => "quotient".to_string(),
        other => other.to_string(),
    }
}

/// Python: class Requirements(object)
#[derive(Debug, Clone)]
pub struct Requirements {
    pub requirements: Vec<String>,
}

impl Requirements {
    pub fn new(requirements: Vec<String>) -> Self {
        Requirements { requirements }
    }

    pub fn has(&self, req: &str) -> bool {
        self.requirements.iter().any(|r| r == req)
    }
}

/// Python: class Task(object)
/// The main PDDL task structure, aggregating everything parsed.
#[derive(Debug, Clone)]
pub struct Task {
    pub domain_name: String,
    pub task_name: String,
    pub requirements: Requirements,
    pub types: Vec<Type>,
    pub objects: Vec<TypedObject>,
    pub predicates: Vec<Predicate>,
    pub functions: Vec<Function>,
    pub init: Vec<Atom>,
    pub num_init: Vec<FunctionAssignment>,
    pub goal: Condition,
    pub actions: Vec<Action>,
    pub axioms: Vec<Axiom>,
    pub metric: (String, PrimitiveNumericExpression),
    pub function_administrator: DerivedFunctionAdministrator,
    pub global_constraint: Condition,
}

impl Task {
    pub fn new(
        domain_name: String,
        task_name: String,
        requirements: Requirements,
        types: Vec<Type>,
        objects: Vec<TypedObject>,
        predicates: Vec<Predicate>,
        functions: Vec<Function>,
        init: Vec<Atom>,
        num_init: Vec<FunctionAssignment>,
        goal: Condition,
        actions: Vec<Action>,
        axioms: Vec<Axiom>,
        metric: (String, PrimitiveNumericExpression),
    ) -> Self {
        // Python: FUNCTION_SYMBOLS computed from functions
        let mut function_admin = DerivedFunctionAdministrator::new();
        // Register all function symbols
        for func in &functions {
            function_admin.function_symbols.insert(func.name.clone());
        }

        Task {
            domain_name,
            task_name,
            requirements,
            types,
            objects,
            predicates,
            functions,
            init,
            num_init,
            goal,
            actions,
            axioms,
            metric,
            function_administrator: function_admin,
            global_constraint: Condition::Truth,
        }
    }

    /// Python: def add_global_constraints(self)
    /// Creates a global constraint axiom from all axioms marked as global constraints.
    pub fn add_global_constraints(&mut self) {
        let mut universal_constraints: Vec<Condition> = vec![];
        for axiom in &mut self.axioms {
            if axiom.is_global_constraint {
                axiom.is_global_constraint = false;
                universal_constraints.push(Condition::UniversalCondition(
                    super::conditions::UniversalCondition::new(
                        axiom.parameters.clone(),
                        vec![axiom.condition.clone()],
                    ),
                ));
            }
        }

        let condition = if universal_constraints.is_empty() {
            Condition::Truth
        } else {
            Condition::Conjunction(Conjunction::new(universal_constraints))
        };
        let axiom = self.add_axiom(format!("new-axiom@{}", self.axioms.len()), vec![], 0, condition);
        self.global_constraint = Condition::Atom(Atom::new(axiom.predicate, vec![]));
    }

    /// Python: def add_axiom(self, name, parameters, num_external, condition)
    pub fn add_axiom(&mut self, name: String, parameters: Vec<TypedObject>, num_external: usize, condition: Condition) -> Atom {
        let args: Vec<String> = parameters[..num_external].iter()
            .map(|p| p.name.clone())
            .collect();
        let effect = Atom::new(name.clone(), args);
        self.predicates.push(Predicate::new(name.clone(), parameters.clone()));
        self.axioms.push(Axiom::new(name, parameters, num_external, condition));
        effect
    }

    /// Python: def dump(self)
    pub fn dump(&self) {
        println!("Task: {} (domain: {})", self.task_name, self.domain_name);
        println!("  {} types", self.types.len());
        println!("  {} objects", self.objects.len());
        println!("  {} predicates", self.predicates.len());
        println!("  {} functions", self.functions.len());
        println!("  {} init facts", self.init.len());
        println!("  {} numeric init", self.num_init.len());
        println!("  goal: {}", self.goal);
        println!("  {} actions", self.actions.len());
        println!("  {} axioms", self.axioms.len());
    }
}

/// Python: class DerivedFunctionAdministrator(object)
/// Manages derived numeric functions (numeric axioms created during instantiation).
#[derive(Debug, Clone)]
pub struct DerivedFunctionAdministrator {
    pub function_symbols: HashSet<String>,
    pub derived_functions: HashMap<DerivedFunctionKey, NumericAxiom>,
    pub axioms: Vec<NumericAxiom>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum DerivedFunctionKey {
    Constant(NumericConstant),
    AdditiveInverse(String),
    Arithmetic(String, Vec<PrimitiveNumericExpression>),
}

impl DerivedFunctionAdministrator {
    pub fn new() -> Self {
        DerivedFunctionAdministrator {
            function_symbols: HashSet::new(),
            derived_functions: HashMap::new(),
            axioms: vec![],
        }
    }

    pub fn get_all_axioms(&self) -> Vec<NumericAxiom> {
        self.derived_functions.values().cloned().collect()
    }

    fn get_default_variables(&self, nr: usize) -> Vec<TypedObject> {
        (0..nr)
            .map(|index| TypedObject::new(&format!("?v{}", index), "object"))
            .collect()
    }

    fn symbol_from_key(&self, key: &DerivedFunctionKey) -> String {
        let addition = match key {
            DerivedFunctionKey::Constant(nc) => format!("{}", nc),
            DerivedFunctionKey::AdditiveInverse(symbol) => {
                format!("{}_{}", prettyprint("-"), prettyprint(symbol))
            }
            DerivedFunctionKey::Arithmetic(op, parts) => {
                let mut tokens = vec![prettyprint(op)];
                for part in parts {
                    tokens.push(prettyprint(&format!("{}", part)));
                }
                tokens.join("_")
            }
        };
        format!("derived!{}", addition)
    }

    /// Python: def get_derived_function(self, expression, fluent_functions)
    /// Gets or creates a derived function for the given expression.
    pub fn get_derived_function(
        &mut self,
        expression: &FunctionalExpression,
        fluent_functions: &HashSet<PrimitiveNumericExpression>,
    ) -> PrimitiveNumericExpression {
        if let FunctionalExpression::PrimitiveNumericExpression(pne) = expression {
            return pne.clone();
        }

        let args = match expression {
            FunctionalExpression::NumericConstant(_) => vec![],
            FunctionalExpression::AdditiveInverse(ai) => {
                self.get_derived_function(&ai.parts[0], fluent_functions).args
            }
            FunctionalExpression::ArithmeticExpression(ae) => {
                let mut subexpressions: Vec<PrimitiveNumericExpression> = ae
                    .parts
                    .iter()
                    .map(|part| self.get_derived_function(part, fluent_functions))
                    .collect();
                if ae.op == "+" || ae.op == "*" {
                    subexpressions.sort_by_key(|p| format!("{}({})", p.symbol, p.args.join(",")));
                }
                subexpressions.into_iter().flat_map(|df| df.args).collect()
            }
            FunctionalExpression::PrimitiveNumericExpression(_) => unreachable!(),
        };

        let key = match expression {
            FunctionalExpression::NumericConstant(nc) => DerivedFunctionKey::Constant(nc.clone()),
            FunctionalExpression::AdditiveInverse(ai) => {
                let subexp = self.get_derived_function(&ai.parts[0], fluent_functions);
                DerivedFunctionKey::AdditiveInverse(subexp.symbol.clone())
            }
            FunctionalExpression::ArithmeticExpression(ae) => {
                let mut subexpressions: Vec<PrimitiveNumericExpression> = ae
                    .parts
                    .iter()
                    .map(|part| self.get_derived_function(part, fluent_functions))
                    .collect();
                if ae.op == "+" || ae.op == "*" {
                    subexpressions.sort_by(|a, b| format!("{}", a).cmp(&format!("{}", b)));
                }
                DerivedFunctionKey::Arithmetic(ae.op.clone(), subexpressions)
            }
            FunctionalExpression::PrimitiveNumericExpression(_) => unreachable!(),
        };

        if let Some(axiom) = self.derived_functions.get(&key) {
            return PrimitiveNumericExpression::with_type(axiom.name.clone(), args, 'D');
        }

        let (name, args, op, parts) = match expression {
            FunctionalExpression::NumericConstant(nc) => {
                let symbol = self.symbol_from_key(&key);
                (
                    symbol,
                    vec![],
                    String::new(),
                    vec![FunctionalExpression::NumericConstant(nc.clone())],
                )
            }
            FunctionalExpression::AdditiveInverse(ai) => {
                let subexp = self.get_derived_function(&ai.parts[0], fluent_functions);
                let args = subexp.args.clone();
                let default_args = self.get_default_variables(args.len());
                let rewritten = FunctionalExpression::PrimitiveNumericExpression(
                    PrimitiveNumericExpression::with_type(
                        subexp.symbol.clone(),
                        default_args.iter().map(|p| p.name.clone()).collect(),
                        'D',
                    ),
                );
                (
                    self.symbol_from_key(&key),
                    args,
                    "-".to_string(),
                    vec![rewritten],
                )
            }
            FunctionalExpression::ArithmeticExpression(ae) => {
                let mut subexpressions: Vec<PrimitiveNumericExpression> = ae
                    .parts
                    .iter()
                    .map(|part| self.get_derived_function(part, fluent_functions))
                    .collect();
                if ae.op == "+" || ae.op == "*" {
                    subexpressions.sort_by_key(|p| format!("{}({})", p.symbol, p.args.join(",")));
                }
                let args: Vec<String> = subexpressions.iter().flat_map(|df| df.args.clone()).collect();
                let default_args = self.get_default_variables(args.len());
                let mut arg_index = 0;
                let mut rewritten_parts = vec![];
                for df in &subexpressions {
                    let end = arg_index + df.args.len();
                    let slice = default_args[arg_index..end].iter().map(|p| p.name.clone()).collect();
                    rewritten_parts.push(FunctionalExpression::PrimitiveNumericExpression(
                        PrimitiveNumericExpression::with_type(df.symbol.clone(), slice, 'D'),
                    ));
                    arg_index = end;
                }
                (
                    self.symbol_from_key(&key),
                    args,
                    ae.op.clone(),
                    rewritten_parts,
                )
            }
            FunctionalExpression::PrimitiveNumericExpression(_) => unreachable!(),
        };

        let parameters = self.get_default_variables(args.len());
        let axiom = NumericAxiom::new(name.clone(), parameters, op, parts);
        self.function_symbols.insert(name.clone());
        self.derived_functions.insert(key, axiom.clone());
        self.axioms = self.get_all_axioms();
        PrimitiveNumericExpression::with_type(name, args, 'D')
    }

    pub fn dump(&self) {
        println!("DerivedFunctionAdministrator:");
        for (key, axiom) in &self.derived_functions {
            println!("  {:?} -> {} ({})", key, axiom.get_head(), axiom);
        }
    }
}

/// Python: def check_atom_consistency(atom, same_truth_value, other_truth_value, atom_is_true)
pub fn check_atom_consistency(
    atom: &Atom,
    same_truth_value: &HashSet<Atom>,
    other_truth_value: &HashSet<Atom>,
    _atom_is_true: bool,
) -> bool {
    if other_truth_value.contains(atom) {
        false
    } else {
        true
    }
}

/// Python: def check_for_duplicates(lst, what_type, what_list)
pub fn check_for_duplicates(lst: &[String], what_type: &str, what_list: &str) {
    let mut seen = HashSet::new();
    for item in lst {
        if !seen.insert(item) {
            eprintln!("Warning: duplicate {} in {}: {}", what_type, what_list, item);
        }
    }
}
