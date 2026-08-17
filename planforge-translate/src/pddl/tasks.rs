use std::collections::{HashMap, HashSet};

use super::actions::Action;
use super::axioms::{Axiom, NumericAxiom};
use super::conditions::{Atom, Condition, Conjunction};
use super::f_expression::{
    FunctionAssignment, FunctionalExpression, NumericConstant, PrimitiveNumericExpression,
};
use super::functions::Function;
use super::pddl_types::{Type, TypedObject};
use super::predicates::Predicate;

fn prettyprint(symbol: &str) -> String {
    match symbol {
        "-" => "difference".to_string(),
        "+" => "sum".to_string(),
        "*" => "product".to_string(),
        "/" => "quotient".to_string(),
        other => other.to_string(),
    }
}

/// The main PDDL task structure, aggregating everything parsed.
///
/// It keeps what the translation reads. The domain and problem names and the
/// `:requirements` list are parsed for their shape and then dropped, because
/// nothing downstream of here consults them.
#[derive(Debug, Clone)]
pub struct Task {
    pub types: Vec<Type>,
    pub objects: Vec<TypedObject>,
    pub predicates: Vec<Predicate>,
    pub init: Vec<Atom>,
    pub num_init: Vec<FunctionAssignment>,
    pub goal: Condition,
    pub actions: Vec<Action>,
    pub axioms: Vec<Axiom>,
    pub metric: (String, PrimitiveNumericExpression),
    pub function_administrator: DerivedFunctionAdministrator,
    pub global_constraint: Condition,
}

/// The objects grounding gives an action's or an axiom's parameters.
///
/// A parameter list is a handful of entries long, so a scan finds one faster
/// than a hash does, and the mapping borrows both the parameter names and the
/// object names rather than copying them: it is rebuilt for every one of the
/// reachable instances, of which a large task has hundreds of thousands.
#[derive(Debug, Default, Clone)]
pub struct VarMapping<'a> {
    bindings: Vec<(&'a str, &'a str)>,
}

impl<'a> VarMapping<'a> {
    /// Binds `parameter` to `object`, replacing whatever it was bound to.
    pub fn bind(&mut self, parameter: &'a str, object: &'a str) {
        match self
            .bindings
            .iter_mut()
            .find(|(name, _)| *name == parameter)
        {
            Some(binding) => binding.1 = object,
            None => self.bindings.push((parameter, object)),
        }
    }

    /// Binds `parameter` to itself unless it is already bound. A parameter of a
    /// conditional effect is quantified inside the effect, so grounding the
    /// action leaves it standing.
    pub fn bind_to_itself(&mut self, parameter: &'a str) {
        if self.get(parameter).is_none() {
            self.bindings.push((parameter, parameter));
        }
    }

    pub fn clear(&mut self) {
        self.bindings.clear();
    }

    pub fn get(&self, parameter: &str) -> Option<&'a str> {
        self.bindings
            .iter()
            .find(|(name, _)| *name == parameter)
            .map(|&(_, object)| object)
    }

    /// The objects a parameter list is bound to, as an argument list.
    pub fn resolve_parameters(&self, parameters: &[TypedObject]) -> Vec<String> {
        use super::Substitution;
        parameters
            .iter()
            .map(|parameter| self.resolve(&parameter.name).to_owned())
            .collect()
    }
}

/// A binding maps the parameters it binds; every other name is an object or
/// another constant, and stands for itself.
impl super::Substitution for VarMapping<'_> {
    fn resolve<'a>(&'a self, name: &'a str) -> &'a str {
        self.get(name).unwrap_or(name)
    }
}

/// The task-wide tables an action or axiom is instantiated against: what
/// holds initially, which facts and functions are fluent, and what the
/// numeric fluents start at.
pub struct GroundingTables<'a> {
    pub init_facts: &'a HashSet<Atom>,
    /// The *atoms* the relaxed-reachability model proved reachable and whose
    /// predicate some effect or axiom writes.
    ///
    /// Deliberately a set of atoms rather than of predicate names: a ground
    /// atom of a fluent predicate can still be unreachable, and then a
    /// condition on it is statically false. Testing the predicate instead would
    /// keep that condition as a fluent one, and it would later be dropped for
    /// having no SAS variable — which turns an inapplicable operator or an
    /// unsupported axiom into an unconditional one.
    pub fluent_facts: &'a HashSet<Atom>,
    pub fluent_functions: &'a HashSet<PrimitiveNumericExpression>,
    pub init_function_vals: &'a HashMap<PrimitiveNumericExpression, f64>,
    /// The objects of each type, including by every supertype they inherit.
    ///
    /// Needed because a universally quantified effect is grounded here rather
    /// than by reachability: `forall(?i - item) (marked ?i)` stands for one
    /// effect per object of type `item`, and only the object universe says which
    /// those are.
    pub objects_by_type: &'a HashMap<String, Vec<String>>,
}

/// What the `(define (domain ...))` form declares.
pub struct DomainDefinition {
    pub types: Vec<Type>,
    pub constants: Vec<TypedObject>,
    pub predicates: Vec<Predicate>,
    pub functions: Vec<Function>,
    pub actions: Vec<Action>,
    pub axioms: Vec<Axiom>,
}

/// What the `(define (problem ...))` form declares.
pub struct ProblemDefinition {
    pub objects: Vec<TypedObject>,
    pub init: Vec<Atom>,
    pub num_init: Vec<FunctionAssignment>,
    pub goal: Condition,
    pub metric: Option<(String, FunctionalExpression)>,
}

impl Task {
    /// Joins the two halves of a PDDL task: the problem's objects extend the
    /// domain's constants, every object is made equal to itself so that `=`
    /// preconditions can be grounded, and the metric defaults to minimising
    /// `total-cost`, which is then declared if the domain did not declare it.
    pub fn new(domain: DomainDefinition, problem: ProblemDefinition) -> Self {
        let mut objects = domain.constants;
        objects.extend(problem.objects);

        let mut init = problem.init;
        init.extend(objects.iter().map(|object| {
            Atom::new(
                "=".to_string(),
                vec![object.name.clone(), object.name.clone()],
            )
        }));

        let mut functions = domain.functions;
        if !functions.iter().any(|f| f.name == "total-cost") {
            functions.push(Function::new(
                "total-cost".to_string(),
                vec![],
                "number".to_string(),
            ));
        }
        let metric_expression = problem.metric.unwrap_or_else(|| {
            (
                "<".to_string(),
                FunctionalExpression::PrimitiveNumericExpression(
                    PrimitiveNumericExpression::with_type("total-cost".to_string(), vec![], 'I'),
                ),
            )
        });

        let mut function_administrator = DerivedFunctionAdministrator::new();
        for function in &functions {
            function_administrator
                .function_symbols
                .insert(function.name.clone());
        }
        let metric = (
            metric_expression.0,
            function_administrator.get_derived_function(&metric_expression.1),
        );

        Task {
            types: domain.types,
            objects,
            predicates: domain.predicates,
            init,
            num_init: problem.num_init,
            goal: problem.goal,
            actions: domain.actions,
            axioms: domain.axioms,
            metric,
            function_administrator,
            global_constraint: Condition::Truth,
        }
    }

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
        let axiom = self.add_axiom(self.fresh_axiom_name(), vec![], 0, condition);
        self.global_constraint = Condition::Atom(Atom::new(axiom.predicate, vec![]));
    }

    /// An axiom name no axiom in the task uses yet.
    ///
    /// Two places invent axioms: the global constraint, and the pass that
    /// replaces a universal condition by the negation of a new derived
    /// predicate. Both used to spell the name themselves, and the global
    /// constraint used `axioms.len()` as if that were a counter. It is not: with
    /// one existing axiom the next index is 1, but so is `new-axiom@1` if the
    /// other generator already took it, and in a task with no axioms both
    /// generators produce `new-axiom@0`.
    ///
    /// Colliding is not a cosmetic problem. Two axioms with one head are read as
    /// two ways of proving it, so a collision with the global constraint, whose
    /// body is `Truth` when there are no global constraints, makes the other
    /// axiom's head unconditionally true. That is silent: the task still
    /// translates, and a `forall` precondition guarded by such a head simply
    /// never holds.
    pub fn fresh_axiom_name(&self) -> String {
        let taken: HashSet<&str> = self.axioms.iter().map(|a| a.name.as_str()).collect();
        (0..)
            .map(|index| format!("new-axiom@{index}"))
            .find(|name| !taken.contains(name.as_str()))
            .expect("the integers are not exhausted")
    }

    pub fn add_axiom(
        &mut self,
        name: String,
        parameters: Vec<TypedObject>,
        num_external: usize,
        condition: Condition,
    ) -> Atom {
        let args: Vec<String> = parameters[..num_external]
            .iter()
            .map(|p| p.name.clone())
            .collect();
        let effect = Atom::new(name.clone(), args);
        self.predicates
            .push(Predicate::new(name.clone(), parameters.clone()));
        self.axioms
            .push(Axiom::new(name, parameters, num_external, condition));
        effect
    }
}

/// Manages derived numeric functions (numeric axioms created during instantiation).
#[derive(Debug, Clone)]
pub struct DerivedFunctionAdministrator {
    pub function_symbols: HashSet<String>,
    derived_functions: HashMap<DerivedFunctionKey, NumericAxiom>,
    derived_functions_by_name: HashMap<String, DerivedFunctionKey>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum DerivedFunctionKey {
    Constant(NumericConstant),
    AdditiveInverse(String),
    Arithmetic(String, Vec<PrimitiveNumericExpression>),
}

impl Default for DerivedFunctionAdministrator {
    fn default() -> Self {
        Self::new()
    }
}

impl DerivedFunctionAdministrator {
    pub fn new() -> Self {
        DerivedFunctionAdministrator {
            function_symbols: HashSet::new(),
            derived_functions: HashMap::new(),
            derived_functions_by_name: HashMap::new(),
        }
    }

    pub fn get_all_axioms(&self) -> Vec<NumericAxiom> {
        self.derived_functions.values().cloned().collect()
    }

    /// The axiom that defines the derived function called `name`.
    ///
    /// A derived function is keyed by the expression it stands for and named
    /// after that key, so a name belongs to at most one of them. Grounding asks
    /// this once per instance of every action, which is why it does not go
    /// through [`Self::get_all_axioms`] and its copy of the whole table.
    pub fn axiom_named(&self, name: &str) -> Option<&NumericAxiom> {
        self.derived_functions_by_name
            .get(name)
            .and_then(|key| self.derived_functions.get(key))
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

    /// Gets or creates a derived function for the given expression.
    /// The numeric variable that stands for `expression`, defining it by a new
    /// numeric axiom if no equivalent expression has one yet.
    pub fn get_derived_function(
        &mut self,
        expression: &FunctionalExpression,
    ) -> PrimitiveNumericExpression {
        if let FunctionalExpression::PrimitiveNumericExpression(pne) = expression {
            return pne.clone();
        }

        let mut subexpressions: Vec<PrimitiveNumericExpression> = match expression {
            FunctionalExpression::NumericConstant(_) => Vec::new(),
            FunctionalExpression::AdditiveInverse(ai) => {
                vec![self.get_derived_function(&ai.parts[0])]
            }
            FunctionalExpression::ArithmeticExpression(ae) => ae
                .parts
                .iter()
                .map(|part| self.get_derived_function(part))
                .collect(),
            FunctionalExpression::PrimitiveNumericExpression(_) => unreachable!(),
        };
        if matches!(expression, FunctionalExpression::ArithmeticExpression(ae) if ae.op == "+" || ae.op == "*")
        {
            sort_commutative_operands(&mut subexpressions);
        }
        let args: Vec<String> = subexpressions
            .iter()
            .flat_map(|derived| derived.args.iter().cloned())
            .collect();

        let key = match expression {
            FunctionalExpression::NumericConstant(nc) => DerivedFunctionKey::Constant(nc.clone()),
            FunctionalExpression::AdditiveInverse(_) => {
                DerivedFunctionKey::AdditiveInverse(subexpressions[0].symbol.clone())
            }
            FunctionalExpression::ArithmeticExpression(ae) => {
                DerivedFunctionKey::Arithmetic(ae.op.clone(), subexpressions.clone())
            }
            FunctionalExpression::PrimitiveNumericExpression(_) => unreachable!(),
        };

        if let Some(axiom) = self.derived_functions.get(&key) {
            return PrimitiveNumericExpression::with_type(axiom.name.clone(), args, 'D');
        }

        let name = self.symbol_from_key(&key);
        let (op, parts) = match expression {
            FunctionalExpression::NumericConstant(nc) => (
                String::new(),
                vec![FunctionalExpression::NumericConstant(nc.clone())],
            ),
            FunctionalExpression::AdditiveInverse(_) => {
                let subexpression = &subexpressions[0];
                let default_args = self.get_default_variables(args.len());
                let rewritten = FunctionalExpression::PrimitiveNumericExpression(
                    PrimitiveNumericExpression::with_type(
                        subexpression.symbol.clone(),
                        default_args.iter().map(|p| p.name.clone()).collect(),
                        'D',
                    ),
                );
                ("-".to_string(), vec![rewritten])
            }
            FunctionalExpression::ArithmeticExpression(ae) => {
                let default_args = self.get_default_variables(args.len());
                let mut arg_index = 0;
                let mut rewritten_parts = vec![];
                for df in &subexpressions {
                    let end = arg_index + df.args.len();
                    let slice = default_args[arg_index..end]
                        .iter()
                        .map(|p| p.name.clone())
                        .collect();
                    rewritten_parts.push(FunctionalExpression::PrimitiveNumericExpression(
                        PrimitiveNumericExpression::with_type(df.symbol.clone(), slice, 'D'),
                    ));
                    arg_index = end;
                }
                (ae.op.clone(), rewritten_parts)
            }
            FunctionalExpression::PrimitiveNumericExpression(_) => unreachable!(),
        };

        let parameters = self.get_default_variables(args.len());
        let axiom = NumericAxiom::new(name.clone(), parameters, op, parts);
        self.function_symbols.insert(name.clone());
        assert!(
            self.derived_functions_by_name
                .insert(name.clone(), key.clone())
                .is_none(),
            "two derived expressions generated the same symbol {name}"
        );
        self.derived_functions.insert(key, axiom);
        PrimitiveNumericExpression::with_type(name, args, 'D')
    }
}

/// Orders the operands of a commutative arithmetic expression so that two
/// expressions differing only in operand order map to the same derived
/// function. The key is formatted once per operand rather than at every
/// comparison.
fn sort_commutative_operands(operands: &mut [PrimitiveNumericExpression]) {
    operands.sort_by_cached_key(|pne| format!("{}({})", pne.symbol, pne.args.join(",")));
}
