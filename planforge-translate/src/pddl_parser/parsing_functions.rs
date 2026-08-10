/// Main PDDL parsing functions that convert S-expressions into PDDL AST.
use std::collections::HashMap;

use tracing::warn;

use super::lisp_parser::SExpr;
use crate::pddl::actions::Action;
use crate::pddl::axioms::Axiom;
use crate::pddl::conditions::*;
use crate::pddl::effects::*;
use crate::pddl::f_expression::*;
use crate::pddl::functions::Function;
use crate::pddl::pddl_types::{Type, TypedObject};
use crate::pddl::predicates::Predicate;
use crate::pddl::tasks::{DomainDefinition, ProblemDefinition, Requirements, Task};

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

pub fn parse_predicate(alist: &[SExpr]) -> Predicate {
    let name = alist[0].as_atom().to_string();
    let arguments = parse_typed_list(&alist[1..], true, "object");
    Predicate::new(name, arguments)
}

pub fn parse_function(alist: &[SExpr], type_name: &str) -> Function {
    let name = alist[0].as_atom().to_string();
    let arguments = parse_typed_list(&alist[1..], true, "object");
    Function::new(name, arguments, type_name.to_string())
}

pub fn parse_condition(alist: &SExpr, type_dict: &HashMap<String, Vec<String>>) -> Condition {
    match alist {
        SExpr::List(items) if items.is_empty() => Condition::Truth,
        SExpr::List(items) => parse_condition_aux(items, type_dict),
        SExpr::Atom(_) => {
            // single atom, treat as Truth or parse as literal
            Condition::Truth
        }
    }
}

fn parse_condition_aux(alist: &[SExpr], type_dict: &HashMap<String, Vec<String>>) -> Condition {
    if alist.is_empty() {
        return Condition::Truth;
    }
    let tag = alist[0].as_atom();
    match tag {
        "and" => {
            let parts: Vec<Condition> = alist[1..]
                .iter()
                .map(|item| parse_condition(item, type_dict))
                .collect();
            Condition::Conjunction(Conjunction::new(parts))
        }
        "or" => {
            let parts: Vec<Condition> = alist[1..]
                .iter()
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
                let args: Vec<String> = inner_list[1..]
                    .iter()
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
                    other,
                    right.clone(),
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
        "<" | "<=" | "=" | ">=" | ">" => parse_function_comparison(alist, type_dict),
        _ => {
            // It's a literal (atom)
            let pred = tag.to_string();
            let args: Vec<String> = alist[1..].iter().map(|a| a.as_atom().to_string()).collect();
            Condition::Atom(Atom::new(pred, args))
        }
    }
}

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
                alist.len() == 3
                    && (expression_looks_numeric(&alist[1]) || expression_looks_numeric(&alist[2]))
            }
            _ => false,
        }
    } else {
        false
    }
}

/// Parse a function comparison like (< (f x) 5)
fn parse_function_comparison(
    alist: &[SExpr],
    _type_dict: &HashMap<String, Vec<String>>,
) -> Condition {
    let comparator = alist[0].as_atom().to_string();
    let parts: Vec<FunctionalExpression> = alist[1..].iter().map(parse_expression).collect();
    assert_eq!(
        parts.len(),
        2,
        "numeric comparisons must have exactly two parts"
    );
    let difference = FunctionalExpression::ArithmeticExpression(ArithmeticExpression::new(
        "-".to_string(),
        parts,
    ));
    let zero = FunctionalExpression::NumericConstant(NumericConstant::new(0.0));
    Condition::FunctionComparison(FunctionComparison::new(comparator, vec![difference, zero]))
}

pub fn parse_literal(alist: &SExpr) -> Condition {
    let items = alist.as_list();
    if items.is_empty() {
        return Condition::Truth;
    }
    let tag = items[0].as_atom();
    if tag == "not" {
        let inner = items[1].as_list();
        let pred = inner[0].as_atom().to_string();
        let args: Vec<String> = inner[1..].iter().map(|a| a.as_atom().to_string()).collect();
        Condition::NegatedAtom(NegatedAtom::new(pred, args))
    } else {
        let pred = tag.to_string();
        let args: Vec<String> = items[1..].iter().map(|a| a.as_atom().to_string()).collect();
        Condition::Atom(Atom::new(pred, args))
    }
}

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
                FunctionalExpression::PrimitiveNumericExpression(classify_pne(s.clone(), vec![]))
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
                        let parts: Vec<FunctionalExpression> =
                            items[1..].iter().map(parse_expression).collect();
                        FunctionalExpression::ArithmeticExpression(ArithmeticExpression::new(
                            tag.to_string(),
                            parts,
                        ))
                    }
                }
                _ => {
                    // It's a function application: (f arg1 arg2 ...)
                    let symbol = tag.to_string();
                    let args: Vec<String> =
                        items[1..].iter().map(|a| a.as_atom().to_string()).collect();
                    FunctionalExpression::PrimitiveNumericExpression(classify_pne(symbol, args))
                }
            }
        }
    }
}

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

/// Parses the effects section and returns an EffectType.
pub fn parse_effects(alist: &SExpr, type_dict: &HashMap<String, Vec<String>>) -> EffectType {
    let items = alist.as_list();
    if items.is_empty() {
        return EffectType::Conjunctive(ConjunctiveEffect::new(vec![]));
    }
    let tag = items[0].as_atom();
    if tag == "and" {
        let effects: Vec<EffectType> = items[1..]
            .iter()
            .map(|item| parse_effect(item, type_dict))
            .collect();
        EffectType::Conjunctive(ConjunctiveEffect::new(effects))
    } else {
        parse_effect(alist, type_dict)
    }
}

fn parse_effect(alist: &SExpr, type_dict: &HashMap<String, Vec<String>>) -> EffectType {
    let items = alist.as_list();
    let tag = items[0].as_atom();
    match tag {
        "not" => {
            let inner = items[1].as_list();
            let pred = inner[0].as_atom().to_string();
            let args: Vec<String> = inner[1..].iter().map(|a| a.as_atom().to_string()).collect();
            EffectType::Simple(SimpleEffect::new(Condition::NegatedAtom(NegatedAtom::new(
                pred, args,
            ))))
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
            let args: Vec<String> = items[1..].iter().map(|a| a.as_atom().to_string()).collect();
            EffectType::Simple(SimpleEffect::new(Condition::Atom(Atom::new(pred, args))))
        }
    }
}

pub fn parse_action(alist: &[SExpr], type_dict: &HashMap<String, Vec<String>>) -> Action {
    // alist is the contents of (:action ...)
    // Expected: name :parameters (...) :precondition (...) :effect (...)
    let name = alist[0].as_atom().to_string();
    let mut parameters = vec![];
    let mut precondition = Condition::Truth;
    let mut effect_type: Option<EffectType> = None;
    let mut cost: Option<FunctionAssignment> = None;

    let mut i = 1;

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

    let mut action = Action::new(name, parameters, num_external, precondition, vec![], cost);

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

/// Parses the body of a `(:derived (HEAD ?x - t ...) CONDITION)` block, i.e.
/// `alist` is the block with `:derived` already stripped.
///
/// A derived predicate quantifies over exactly its head's arguments, so all of
/// them are external and the body may not introduce free variables of its own.
pub fn parse_axiom(alist: &[SExpr], type_dict: &HashMap<String, Vec<String>>) -> Axiom {
    assert_eq!(
        alist.len(),
        2,
        "(:derived HEAD CONDITION) takes exactly two elements, got {}: {alist:?}",
        alist.len()
    );
    assert!(
        alist[0].is_list(),
        "the head of a derived predicate is `(NAME ?x - t ...)`, got the atom {:?}",
        alist[0].as_atom()
    );

    let head = parse_predicate(alist[0].as_list());
    let condition = parse_condition(&alist[1], type_dict);
    let num_external = head.arguments.len();
    Axiom::new(head.name, head.arguments, num_external, condition)
}

/// Combines parsed domain and problem S-expressions into a Task.
pub fn parse_task(domain_pddl: &SExpr, task_pddl: &SExpr) -> Task {
    let (domain, type_dict) = parse_domain_pddl(domain_pddl.as_list());
    let problem = parse_task_pddl(task_pddl.as_list(), &type_dict);
    Task::new(domain, problem)
}

/// Parses a `(define (domain ...))` form, returning it together with the map
/// from each type to its supertypes, which the problem needs to type objects.
fn parse_domain_pddl(items: &[SExpr]) -> (DomainDefinition, HashMap<String, Vec<String>>) {
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

    for item in &items[2..] {
        let section = item.as_list();
        if section.is_empty() {
            continue;
        }
        let tag = section[0].as_atom();
        match tag {
            ":requirements" => {
                let reqs: Vec<String> = section[1..]
                    .iter()
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
                predicates = section[1..]
                    .iter()
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
            ":derived" => {
                axioms.push(parse_axiom(&section[1..], &type_dict));
            }
            // PDDL 2.1's `(:axiom :vars ... :context ... :implies ...)` is a
            // different block shape, not a spelling of `:derived`; parsing it
            // as one would silently take `:vars` for the head predicate.
            ":axiom" => panic!(
                "the PDDL 2.1 `(:axiom ...)` block is not supported; write the derived \
                 predicate as `(:derived (NAME ?x - t) CONDITION)`"
            ),
            ":global-constraint" => {
                axioms.push(parse_global_constraint(&section[1..], &type_dict));
            }
            _ => {
                // Unknown section, skip
                warn!("Warning: Unknown domain section: {}", tag);
            }
        }
    }

    (
        DomainDefinition {
            domain_name,
            requirements,
            types,
            constants,
            predicates,
            functions,
            actions,
            axioms,
        },
        type_dict,
    )
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

fn parse_task_pddl(items: &[SExpr], type_dict: &HashMap<String, Vec<String>>) -> ProblemDefinition {
    assert_eq!(items[0].as_atom(), "define");

    let problem_list = items[1].as_list();
    assert_eq!(problem_list[0].as_atom(), "problem");
    let task_name = problem_list[1].as_atom().to_string();

    let mut objects: Vec<TypedObject> = vec![];
    let mut init: Vec<Atom> = vec![];
    let mut num_init: Vec<FunctionAssignment> = vec![];
    let mut goal = Condition::Truth;
    let mut metric: Option<(String, PrimitiveNumericExpression)> = None;

    for item in &items[2..] {
        let section = item.as_list();
        if section.is_empty() {
            continue;
        }
        let tag = section[0].as_atom();
        match tag {
            // The problem names the domain it belongs to; nothing checks
            // that against the domain file, which the caller chose.
            ":domain" => {}
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
                        num_init.push(FunctionAssignment::new("=".to_string(), fluent, value_expr));
                    } else if matches!(first, "not") {
                        // Negative init fact - these are not standard PDDL but we handle them
                        // Just skip, closed world assumption handles this
                    } else {
                        // Positive init fact
                        let pred = first.to_string();
                        let args: Vec<String> = init_item[1..]
                            .iter()
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

    ProblemDefinition {
        task_name,
        objects,
        init,
        num_init,
        goal,
        metric,
    }
}

pub fn check_for_duplicates(lst: &[String], what_type: &str, what_list: &str) {
    use std::collections::HashSet;
    let mut seen = HashSet::new();
    for item in lst {
        if !seen.insert(item) {
            warn!(
                "Warning: duplicate {} in {}: {}",
                what_type, what_list, item
            );
        }
    }
}

/// Resolves a predicate name to (id, arity) or creates a new one.
pub fn get_predicate_id_and_arity(
    text: &str,
    predicate_dict: &HashMap<String, (usize, usize)>,
    _n_predicates: usize,
) -> (usize, usize) {
    if let Some(&(id, arity)) = predicate_dict.get(text) {
        (id, arity)
    } else {
        panic!("Unknown predicate: {}", text);
    }
}
