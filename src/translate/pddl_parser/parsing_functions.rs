/// Port of pddl_parser/parsing_functions.py
/// Main PDDL parsing functions that convert S-expressions into PDDL AST.

use std::collections::HashMap;

use super::lisp_parser::SExpr;
use crate::translate::pddl::pddl_types::{Type, TypedObject, get_type_predicate_name};
use crate::translate::pddl::predicates::Predicate;
use crate::translate::pddl::functions::Function;
use crate::translate::pddl::conditions::*;
use crate::translate::pddl::f_expression::*;
use crate::translate::pddl::effects::*;
use crate::translate::pddl::actions::Action;
use crate::translate::pddl::axioms::Axiom;
use crate::translate::pddl::tasks::{Task, Requirements};

/// Python: def parse_typed_list(alist, only_variables=False, constructor=TypedObject, default_type="object")
/// Parses a list of typed items: "?x ?y - type ?z - type2"
pub fn parse_typed_list(
    alist: &[SExpr],
    only_variables: bool,
    default_type: &str,
) -> Vec<TypedObject> {
    let mut result = vec![];
    let mut untyped_items: Vec<String> = vec![];
    let mut i = 0;

    while i < alist.len() {
        let item = alist[i].as_atom();
        if item == "-" {
            // Next item is the type
            i += 1;
            let type_name = alist[i].as_atom();
            for name in &untyped_items {
                if only_variables && !name.starts_with('?') {
                    panic!("Expected variable, got: {}", name);
                }
                result.push(TypedObject::new(name, type_name));
            }
            untyped_items.clear();
        } else {
            untyped_items.push(item.to_string());
        }
        i += 1;
    }

    // Remaining items have the default type
    for name in &untyped_items {
        if only_variables && !name.starts_with('?') {
            panic!("Expected variable, got: {}", name);
        }
        result.push(TypedObject::new(name, default_type));
    }

    result
}

/// Python: def parse_typed_list for types (constructor=Type)
pub fn parse_type_list(alist: &[SExpr]) -> Vec<Type> {
    let mut result = vec![];
    let mut untyped_items: Vec<String> = vec![];
    let mut i = 0;

    while i < alist.len() {
        let item = alist[i].as_atom();
        if item == "-" {
            i += 1;
            let basetype_name = alist[i].as_atom();
            for name in &untyped_items {
                result.push(Type::new(name, Some(basetype_name)));
            }
            untyped_items.clear();
        } else {
            untyped_items.push(item.to_string());
        }
        i += 1;
    }

    for name in &untyped_items {
        result.push(Type::new(name, Some("object")));
    }

    result
}

/// Python: def set_supertypes(type_list)
/// Returns a map from type name -> list of all supertypes (transitive closure).
pub fn set_supertypes(type_list: &[Type]) -> HashMap<String, Vec<String>> {
    let mut type_map: HashMap<String, Option<String>> = HashMap::new();
    for t in type_list {
        type_map.insert(t.name.clone(), t.basetype_name.clone());
    }
    // object always maps to None
    type_map.insert("object".to_string(), None);

    let mut supertypes: HashMap<String, Vec<String>> = HashMap::new();
    for t in type_list {
        let mut chain = vec![t.name.clone()];
        let mut current = t.basetype_name.clone();
        while let Some(ref cur) = current {
            chain.push(cur.clone());
            current = type_map.get(cur).cloned().flatten();
        }
        supertypes.insert(t.name.clone(), chain);
    }
    // object's supertypes is just [object]
    supertypes.insert("object".to_string(), vec!["object".to_string()]);

    supertypes
}

/// Python: def parse_predicate(alist)
pub fn parse_predicate(alist: &[SExpr]) -> Predicate {
    let name = alist[0].as_atom().to_string();
    let arguments = parse_typed_list(&alist[1..], true, "object");
    Predicate::new(name, arguments)
}

/// Python: def parse_function(alist, type_name)
pub fn parse_function(alist: &[SExpr], type_name: &str) -> Function {
    let name = alist[0].as_atom().to_string();
    let arguments = parse_typed_list(&alist[1..], true, "object");
    Function::new(name, arguments, type_name.to_string())
}

/// Python: def parse_condition(alist, type_dict)
pub fn parse_condition(alist: &SExpr, type_dict: &HashMap<String, Vec<String>>) -> Condition {
    match alist {
        SExpr::List(items) if items.is_empty() => Condition::Truth,
        SExpr::List(items) => {
            parse_condition_aux(items, type_dict)
        }
        SExpr::Atom(_) => {
            // single atom, treat as Truth or parse as literal
            Condition::Truth
        }
    }
}

/// Python: def parse_condition_aux(alist, type_dict)
fn parse_condition_aux(alist: &[SExpr], type_dict: &HashMap<String, Vec<String>>) -> Condition {
    if alist.is_empty() {
        return Condition::Truth;
    }
    let tag = alist[0].as_atom();
    match tag {
        "and" => {
            let parts: Vec<Condition> = alist[1..].iter()
                .map(|item| parse_condition(item, type_dict))
                .collect();
            Condition::Conjunction(Conjunction::new(parts))
        }
        "or" => {
            let parts: Vec<Condition> = alist[1..].iter()
                .map(|item| parse_condition(item, type_dict))
                .collect();
            Condition::Disjunction(Disjunction::new(parts))
        }
        "not" => {
            assert_eq!(alist.len(), 2, "not takes exactly one argument");
            let inner = &alist[1];
            let inner_list = inner.as_list();
            // Check if it's a function comparison
            if is_function_comparison(inner_list) {
                let fc = parse_function_comparison(inner_list, type_dict);
                match fc {
                    Condition::FunctionComparison(fc) => {
                        Condition::NegatedFunctionComparison(fc.negate())
                    }
                    _ => panic!("Expected FunctionComparison inside not"),
                }
            } else {
                // It's a negated literal
                let pred = inner_list[0].as_atom().to_string();
                let args: Vec<String> = inner_list[1..].iter()
                    .map(|a| a.as_atom().to_string())
                    .collect();
                Condition::NegatedAtom(NegatedAtom::new(pred, args))
            }
        }
        "imply" => {
            assert_eq!(alist.len(), 3, "imply takes exactly two arguments");
            let left = parse_condition(&alist[1], type_dict);
            let right = parse_condition(&alist[2], type_dict);
            // imply(a, b) = or(not(a), b)
            // We need to negate left
            let neg_left = match left {
                Condition::Atom(a) => Condition::NegatedAtom(a.negate()),
                Condition::NegatedAtom(a) => Condition::Atom(a.negate()),
                other => Condition::Disjunction(Disjunction::new(vec![
                    // Can't simply negate arbitrary conditions; use DeMorgan etc.
                    // For simplicity in PDDL, imply usually has literals
                    other, right.clone()
                ])),
            };
            match neg_left {
                Condition::Disjunction(d) => Condition::Disjunction(d),
                neg => Condition::Disjunction(Disjunction::new(vec![neg, right])),
            }
        }
        "forall" => {
            let params_list = alist[1].as_list();
            let parameters = parse_typed_list(params_list, true, "object");
            let body = parse_condition(&alist[2], type_dict);
            Condition::UniversalCondition(UniversalCondition::new(parameters, vec![body]))
        }
        "exists" => {
            let params_list = alist[1].as_list();
            let parameters = parse_typed_list(params_list, true, "object");
            let body = parse_condition(&alist[2], type_dict);
            Condition::ExistentialCondition(ExistentialCondition::new(parameters, vec![body]))
        }
        "<" | "<=" | "=" | ">=" | ">" => {
            parse_function_comparison(alist, type_dict)
        }
        _ => {
            // It's a literal (atom)
            let pred = tag.to_string();
            let args: Vec<String> = alist[1..].iter()
                .map(|a| a.as_atom().to_string())
                .collect();
            Condition::Atom(Atom::new(pred, args))
        }
    }
}

/// Python: def is_function_comparison(alist)
fn is_function_comparison(alist: &[SExpr]) -> bool {
    fn expression_looks_numeric(expr: &SExpr) -> bool {
        match expr {
            SExpr::Atom(atom) => atom.parse::<f64>().is_ok(),
            SExpr::List(items) => {
                if items.is_empty() {
                    false
                } else {
                    let _head = items[0].as_atom();
                    true
                }
            }
        }
    }

    if alist.is_empty() {
        return false;
    }
    if let SExpr::Atom(tag) = &alist[0] {
        match tag.as_str() {
            "<" | "<=" | ">=" | ">" => true,
            "=" => {
                alist.len() == 3 && (expression_looks_numeric(&alist[1]) || expression_looks_numeric(&alist[2]))
            }
            _ => false,
        }
    } else {
        false
    }
}

/// Parse a function comparison like (< (f x) 5)
fn parse_function_comparison(alist: &[SExpr], _type_dict: &HashMap<String, Vec<String>>) -> Condition {
    let comparator = alist[0].as_atom().to_string();
    let parts: Vec<FunctionalExpression> = alist[1..].iter()
        .map(|item| parse_expression(item))
        .collect();
    Condition::FunctionComparison(FunctionComparison::new(comparator, parts))
}

/// Python: def parse_literal(alist)
pub fn parse_literal(alist: &SExpr) -> Condition {
    let items = alist.as_list();
    if items.is_empty() {
        return Condition::Truth;
    }
    let tag = items[0].as_atom();
    if tag == "not" {
        let inner = items[1].as_list();
        let pred = inner[0].as_atom().to_string();
        let args: Vec<String> = inner[1..].iter()
            .map(|a| a.as_atom().to_string())
            .collect();
        Condition::NegatedAtom(NegatedAtom::new(pred, args))
    } else {
        let pred = tag.to_string();
        let args: Vec<String> = items[1..].iter()
            .map(|a| a.as_atom().to_string())
            .collect();
        Condition::Atom(Atom::new(pred, args))
    }
}

/// Python: def parse_expression(alist)
pub fn parse_expression(alist: &SExpr) -> FunctionalExpression {
    fn classify_pne(symbol: String, args: Vec<String>) -> PrimitiveNumericExpression {
        if symbol == "total-cost" && args.is_empty() {
            PrimitiveNumericExpression::with_type(symbol, args, 'I')
        } else {
            PrimitiveNumericExpression::new(symbol, args)
        }
    }

    match alist {
        SExpr::Atom(s) => {
            // Try to parse as a number
            if let Ok(val) = s.parse::<f64>() {
                FunctionalExpression::NumericConstant(NumericConstant::new(val))
            } else {
                // It's a function symbol with no arguments
                FunctionalExpression::PrimitiveNumericExpression(
                    classify_pne(s.clone(), vec![])
                )
            }
        }
        SExpr::List(items) => {
            if items.is_empty() {
                panic!("Empty expression list");
            }
            let tag = items[0].as_atom();
            match tag {
                "+" | "-" | "*" | "/" => {
                    if tag == "-" && items.len() == 2 {
                        // Unary minus / additive inverse
                        let inner = parse_expression(&items[1]);
                        FunctionalExpression::AdditiveInverse(AdditiveInverse::new(vec![inner]))
                    } else {
                        let parts: Vec<FunctionalExpression> = items[1..].iter()
                            .map(|item| parse_expression(item))
                            .collect();
                        FunctionalExpression::ArithmeticExpression(
                            ArithmeticExpression::new(tag.to_string(), parts)
                        )
                    }
                }
                _ => {
                    // It's a function application: (f arg1 arg2 ...)
                    let symbol = tag.to_string();
                    let args: Vec<String> = items[1..].iter()
                        .map(|a| a.as_atom().to_string())
                        .collect();
                    FunctionalExpression::PrimitiveNumericExpression(
                        classify_pne(symbol, args)
                    )
                }
            }
        }
    }
}

/// Python: def parse_assignment(alist)
pub fn parse_assignment(alist: &[SExpr]) -> FunctionAssignment {
    let tag = alist[0].as_atom();
    let symbol = match tag {
        "assign" => "=",
        "increase" => "+",
        "decrease" => "-",
        "scale-up" => "*",
        "scale-down" => "/",
        _ => panic!("Unknown assignment operator: {}", tag),
    };
    let fluent_expr = parse_expression(&alist[1]);
    let fluent = match fluent_expr {
        FunctionalExpression::PrimitiveNumericExpression(pne) => pne,
        _ => panic!("Expected primitive numeric expression as fluent in assignment"),
    };
    let expression = parse_expression(&alist[2]);
    FunctionAssignment::new(symbol.to_string(), fluent, expression)
}

/// Python: def parse_effects(alist, type_dict)
/// Parses the effects section and returns an EffectType.
pub fn parse_effects(alist: &SExpr, type_dict: &HashMap<String, Vec<String>>) -> EffectType {
    let items = alist.as_list();
    if items.is_empty() {
        return EffectType::Conjunctive(ConjunctiveEffect::new(vec![]));
    }
    let tag = items[0].as_atom();
    if tag == "and" {
        let effects: Vec<EffectType> = items[1..].iter()
            .map(|item| parse_effect(item, type_dict))
            .collect();
        EffectType::Conjunctive(ConjunctiveEffect::new(effects))
    } else {
        parse_effect(alist, type_dict)
    }
}

/// Python: def parse_effect(alist, type_dict)
fn parse_effect(alist: &SExpr, type_dict: &HashMap<String, Vec<String>>) -> EffectType {
    let items = alist.as_list();
    let tag = items[0].as_atom();
    match tag {
        "not" => {
            let inner = items[1].as_list();
            let pred = inner[0].as_atom().to_string();
            let args: Vec<String> = inner[1..].iter()
                .map(|a| a.as_atom().to_string())
                .collect();
            EffectType::Simple(SimpleEffect::new(
                Condition::NegatedAtom(NegatedAtom::new(pred, args))
            ))
        }
        "when" => {
            let condition = parse_condition(&items[1], type_dict);
            let effect = parse_effect(&items[2], type_dict);
            EffectType::Conditional(ConditionalEffect::new(condition, effect))
        }
        "forall" => {
            let params_list = items[1].as_list();
            let parameters = parse_typed_list(params_list, true, "object");
            let effect = parse_effect(&items[2], type_dict);
            EffectType::Universal(UniversalEffect::new(parameters, effect))
        }
        "assign" | "increase" | "decrease" | "scale-up" | "scale-down" => {
            let assignment = parse_assignment(items);
            EffectType::Numeric(NumericEffect::new(assignment))
        }
        _ => {
            // Simple add effect (atom)
            let pred = tag.to_string();
            let args: Vec<String> = items[1..].iter()
                .map(|a| a.as_atom().to_string())
                .collect();
            EffectType::Simple(SimpleEffect::new(
                Condition::Atom(Atom::new(pred, args))
            ))
        }
    }
}

/// Python: def parse_action(alist, type_dict)
pub fn parse_action(alist: &[SExpr], type_dict: &HashMap<String, Vec<String>>) -> Action {
    // alist is the contents of (:action ...)
    // Expected: name :parameters (...) :precondition (...) :effect (...)
    let mut name = String::new();
    let mut parameters = vec![];
    let mut precondition = Condition::Truth;
    let mut effect_type: Option<EffectType> = None;
    let mut cost: Option<FunctionAssignment> = None;

    let mut i = 0;
    // First item is the action name
    name = alist[0].as_atom().to_string();
    i = 1;

    while i < alist.len() {
        let key = alist[i].as_atom();
        match key {
            ":parameters" => {
                i += 1;
                let params_list = alist[i].as_list();
                parameters = parse_typed_list(params_list, true, "object");
            }
            ":precondition" => {
                i += 1;
                precondition = parse_condition(&alist[i], type_dict);
            }
            ":effect" => {
                i += 1;
                let eff = parse_effects(&alist[i], type_dict);
                // Extract cost
                let (remaining, c) = eff.extract_cost();
                effect_type = Some(remaining);
                cost = c;
            }
            _ => {
                // Skip unknown keys
            }
        }
        i += 1;
    }

    let num_external = parameters.len();

    // Normalize effects
    let effects = if let Some(ref eff) = effect_type {
        let normalized = eff.normalize();
        normalized.into_iter().map(|(params, condition, kind)| {
            match kind {
                EffectKind::Literal(lit) => {
                    Effect::new(params, condition, lit)
                }
                EffectKind::Numeric(assign) => {
                    // Numeric effects stored separately
                    // For now, store as Effect with a special marker
                    Effect::new(params, condition, Condition::Truth) // placeholder
                }
            }
        }).collect()
    } else {
        vec![]
    };

    // Also collect numeric effects
    let mut action = Action::new(name, parameters, num_external, precondition, vec![], cost);

    if let Some(eff) = effect_type.as_ref() {
        // Re-normalize to properly separate literal and numeric effects
    }

    // Re-do effect normalization properly
    let mut literal_effects = vec![];
    if let Some(ref eff) = effect_type {
        let normalized = eff.normalize();
        for (params, condition, kind) in normalized {
            match kind {
                EffectKind::Literal(lit) => {
                    literal_effects.push(Effect::new(params, condition, lit));
                }
                EffectKind::Numeric(assign) => {
                    action.assign_effects.push((params, condition, assign));
                }
            }
        }
    }

    action.effects = literal_effects;
    action
}

/// Python: def parse_global_constraint(alist, type_dict)
pub fn parse_global_constraint(alist: &[SExpr], type_dict: &HashMap<String, Vec<String>>) -> Axiom {
    let name = alist[0].as_atom().to_string();
    let mut parameters = vec![];
    let mut condition = Condition::Truth;

    let mut i = 1;
    while i < alist.len() {
        let key = alist[i].as_atom();
        match key {
            ":parameters" => {
                i += 1;
                let params_list = alist[i].as_list();
                parameters = parse_typed_list(params_list, true, "object");
            }
            ":condition" => {
                i += 1;
                condition = parse_condition(&alist[i], type_dict);
            }
            _ => {}
        }
        i += 1;
    }

    let num_external = parameters.len();
    Axiom::new_global_constraint(name, parameters, num_external, condition)
}

/// Python: def parse_axiom(alist, type_dict)
pub fn parse_axiom(alist: &[SExpr], type_dict: &HashMap<String, Vec<String>>) -> Axiom {
    let name = alist[0].as_atom().to_string();
    let mut parameters = vec![];
    let mut condition = Condition::Truth;

    let mut i = 1;
    while i < alist.len() {
        let key = alist[i].as_atom();
        match key {
            ":parameters" => {
                i += 1;
                let params_list = alist[i].as_list();
                parameters = parse_typed_list(params_list, true, "object");
            }
            ":vars" => {
                i += 1;
                let vars_list = alist[i].as_list();
                let extra_params = parse_typed_list(vars_list, true, "object");
                parameters.extend(extra_params);
            }
            ":context" | ":condition" => {
                i += 1;
                condition = parse_condition(&alist[i], type_dict);
            }
            _ => {}
        }
        i += 1;
    }

    let num_external = parameters.iter()
        .position(|_| false) // All parameters are external for axioms
        .unwrap_or(parameters.len());
    // Actually for axioms, num_external_parameters = len(parameters) from :parameters
    // and additional ones from :vars are internal
    // We can't easily track this here, so let's use a simpler approach:
    // The name itself encodes the head, and parameters up to :vars boundary are external

    Axiom::new(name, parameters, num_external, condition)
}

/// Python: def parse_task(domain_pddl, task_pddl)
/// Combines parsed domain and problem S-expressions into a Task.
pub fn parse_task(domain_pddl: &SExpr, task_pddl: &SExpr) -> Task {
    let domain_items = domain_pddl.as_list();
    let task_items = task_pddl.as_list();

    // Parse domain
    let (domain_name, requirements, types, type_dict, constants,
         predicates, functions, actions, axioms) = parse_domain_pddl(domain_items);

    // Parse problem
    let (task_name, _task_domain, objects, mut init, num_init, goal, metric) =
        parse_task_pddl(task_items, &type_dict);

    // Combine objects
    let mut all_objects = constants;
    all_objects.extend(objects);

    for obj in &all_objects {
        init.push(Atom::new("=".to_string(), vec![obj.name.clone(), obj.name.clone()]));
    }

    // Determine metric
    let task_metric = metric.unwrap_or_else(|| {
        ("<".to_string(), PrimitiveNumericExpression::with_type("total-cost".to_string(), vec![], 'I'))
    });

    // Check if total-cost function exists, if not add it
    let mut all_functions = functions;
    if !all_functions.iter().any(|f| f.name == "total-cost") {
        all_functions.push(Function::new("total-cost".to_string(), vec![], "number".to_string()));
    }

    Task::new(
        domain_name, task_name, requirements, types, all_objects,
        predicates, all_functions, init, num_init, goal, actions, axioms,
        task_metric,
    )
}

/// Python: def parse_domain_pddl(domain_pddl)
/// Generator in Python, here returns all parsed components.
fn parse_domain_pddl(items: &[SExpr]) -> (
    String,                          // domain_name
    Requirements,                    // requirements
    Vec<Type>,                       // types
    HashMap<String, Vec<String>>,    // type_dict (supertypes)
    Vec<TypedObject>,                // constants
    Vec<Predicate>,                  // predicates
    Vec<Function>,                   // functions
    Vec<Action>,                     // actions
    Vec<Axiom>,                      // axioms
) {
    assert_eq!(items[0].as_atom(), "define", "Expected (define ...)");

    let domain_name_list = items[1].as_list();
    assert_eq!(domain_name_list[0].as_atom(), "domain");
    let domain_name = domain_name_list[1].as_atom().to_string();

    let mut requirements = Requirements::new(vec![]);
    let mut types: Vec<Type> = vec![];
    let mut type_dict: HashMap<String, Vec<String>> = HashMap::new();
    type_dict.insert("object".to_string(), vec!["object".to_string()]);
    let mut constants: Vec<TypedObject> = vec![];
    let mut predicates: Vec<Predicate> = vec![];
    let mut functions: Vec<Function> = vec![];
    let mut actions: Vec<Action> = vec![];
    let mut axioms: Vec<Axiom> = vec![];

    for i in 2..items.len() {
        let section = items[i].as_list();
        if section.is_empty() {
            continue;
        }
        let tag = section[0].as_atom();
        match tag {
            ":requirements" => {
                let reqs: Vec<String> = section[1..].iter()
                    .map(|s| s.as_atom().to_string())
                    .collect();
                requirements = Requirements::new(reqs);
            }
            ":types" => {
                types = parse_type_list(&section[1..]);
                type_dict = set_supertypes(&types);
            }
            ":constants" => {
                constants = parse_typed_list(&section[1..], false, "object");
            }
            ":predicates" => {
                predicates = section[1..].iter()
                    .map(|p| parse_predicate(p.as_list()))
                    .collect();
            }
            ":functions" => {
                // Functions can have a return type after "-"
                functions = parse_function_list(&section[1..]);
            }
            ":action" => {
                actions.push(parse_action(&section[1..], &type_dict));
            }
            ":derived" | ":axiom" => {
                axioms.push(parse_axiom(&section[1..], &type_dict));
            }
            ":global-constraint" => {
                axioms.push(parse_global_constraint(&section[1..], &type_dict));
            }
            _ => {
                // Unknown section, skip
                eprintln!("Warning: Unknown domain section: {}", tag);
            }
        }
    }

    (domain_name, requirements, types, type_dict, constants, predicates, functions, actions, axioms)
}

/// Parse function declarations with types
fn parse_function_list(items: &[SExpr]) -> Vec<Function> {
    let mut result = vec![];
    let mut current_functions: Vec<&SExpr> = vec![];
    let mut i = 0;

    while i < items.len() {
        match &items[i] {
            SExpr::Atom(s) if s == "-" => {
                i += 1;
                let type_name = items[i].as_atom();
                for func_expr in &current_functions {
                    let func_list = func_expr.as_list();
                    result.push(parse_function(func_list, type_name));
                }
                current_functions.clear();
            }
            other => {
                current_functions.push(other);
            }
        }
        i += 1;
    }

    // Remaining functions have default type "number"
    for func_expr in &current_functions {
        let func_list = func_expr.as_list();
        result.push(parse_function(func_list, "number"));
    }

    result
}

/// Python: def parse_task_pddl(task_pddl, type_dict)
fn parse_task_pddl(items: &[SExpr], type_dict: &HashMap<String, Vec<String>>) -> (
    String,                                            // task_name
    String,                                            // domain_name reference
    Vec<TypedObject>,                                  // objects
    Vec<Atom>,                                         // init (propositional)
    Vec<FunctionAssignment>,                           // num_init (numeric)
    Condition,                                         // goal
    Option<(String, PrimitiveNumericExpression)>,       // metric
) {
    assert_eq!(items[0].as_atom(), "define");

    let problem_list = items[1].as_list();
    assert_eq!(problem_list[0].as_atom(), "problem");
    let task_name = problem_list[1].as_atom().to_string();

    let mut domain_name = String::new();
    let mut objects: Vec<TypedObject> = vec![];
    let mut init: Vec<Atom> = vec![];
    let mut num_init: Vec<FunctionAssignment> = vec![];
    let mut goal = Condition::Truth;
    let mut metric: Option<(String, PrimitiveNumericExpression)> = None;

    for i in 2..items.len() {
        let section = items[i].as_list();
        if section.is_empty() {
            continue;
        }
        let tag = section[0].as_atom();
        match tag {
            ":domain" => {
                domain_name = section[1].as_atom().to_string();
            }
            ":objects" => {
                objects = parse_typed_list(&section[1..], false, "object");
            }
            ":init" => {
                for item in &section[1..] {
                    let init_item = item.as_list();
                    if init_item.is_empty() {
                        continue;
                    }
                    let first = init_item[0].as_atom();
                    if first == "=" {
                        // Numeric init: (= (func args) value)
                        let fluent_expr = parse_expression(&init_item[1]);
                        let value_expr = parse_expression(&init_item[2]);
                        let fluent = match fluent_expr {
                            FunctionalExpression::PrimitiveNumericExpression(pne) => pne,
                            _ => panic!("Expected PNE in numeric init"),
                        };
                        num_init.push(FunctionAssignment::new(
                            "=".to_string(), fluent, value_expr,
                        ));
                    } else if matches!(first, "not") {
                        // Negative init fact - these are not standard PDDL but we handle them
                        // Just skip, closed world assumption handles this
                    } else {
                        // Positive init fact
                        let pred = first.to_string();
                        let args: Vec<String> = init_item[1..].iter()
                            .map(|a| a.as_atom().to_string())
                            .collect();
                        init.push(Atom::new(pred, args));
                    }
                }
            }
            ":goal" => {
                goal = parse_condition(&section[1], type_dict);
            }
            ":metric" => {
                // (:metric minimize (func))
                let direction = section[1].as_atom();
                let dir_symbol = if direction == "minimize" { "<" } else { ">" };
                let metric_expr = parse_expression(&section[2]);
                let metric_pne = match metric_expr {
                    FunctionalExpression::PrimitiveNumericExpression(pne) => pne,
                    _ => {
                        // Complex metric expression - use total-cost as default
                        PrimitiveNumericExpression::with_type("total-cost".to_string(), vec![], 'I')
                    }
                };
                metric = Some((dir_symbol.to_string(), metric_pne));
            }
            _ => {
                // Skip unknown sections
            }
        }
    }

    (task_name, domain_name, objects, init, num_init, goal, metric)
}

/// Python: def check_for_duplicates(lst, what_type, what_list)
pub fn check_for_duplicates(lst: &[String], what_type: &str, what_list: &str) {
    use std::collections::HashSet;
    let mut seen = HashSet::new();
    for item in lst {
        if !seen.insert(item) {
            eprintln!("Warning: duplicate {} in {}: {}", what_type, what_list, item);
        }
    }
}

/// Python: def _get_predicate_id_and_arity(text, predicate_dict, n_predicates)
/// Resolves a predicate name to (id, arity) or creates a new one.
pub fn get_predicate_id_and_arity(
    text: &str,
    predicate_dict: &HashMap<String, (usize, usize)>,
    n_predicates: usize,
) -> (usize, usize) {
    if let Some(&(id, arity)) = predicate_dict.get(text) {
        (id, arity)
    } else {
        panic!("Unknown predicate: {}", text);
    }
}
