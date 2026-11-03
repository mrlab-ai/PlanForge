//! The rules of the grounding model: for every action, axiom and numeric
//! expression of the normalized task, the Datalog rule that makes it
//! reachable, and the predicate names those rules use.

use std::collections::{HashMap, HashSet};

use crate::pddl::conditions::*;
use crate::pddl::f_expression::*;
use crate::pddl::pddl_types::TypedObject;
use crate::pddl::tasks::Task;

// ==================== Exploration rules ====================

/// Builds a set of rules for the grounding process.
/// These rules encode what atoms are reachable.
pub fn build_exploration_rules(task: &Task) -> Vec<ExplorationRule> {
    let mut rules = vec![];

    // Action applicability rules.
    for action in &task.actions {
        // An action whose cost is a static function of its parameters is only
        // reachable where that function has a value. Fast Downward does the same,
        // by handing the cost's expression to `condition_to_rule_body` so the rule
        // body carries its `defined!` guard.
        //
        // Without the guard, grounding produced instances whose cost is undefined:
        // `elevators` declares `(travel-slow ?f1 ?f2)` and gives it a value only
        // for the floor pairs that can actually be travelled, so `(move-up-slow
        // ?lift ?f ?f)` was generated with no cost to look up. Refusing the task
        // was the visible symptom; the quiet version of the same bug would be an
        // operator whose cost is whatever the lookup happened to return.
        let mut conditions = condition_to_rule_body(&action.parameters, &action.precondition);
        if let Some(cost) = &action.cost
            && let FunctionalExpression::PrimitiveNumericExpression(pne) = &cost.expression
            && !pne.args.is_empty()
        {
            conditions.push(Condition::Atom(Atom::new(
                get_function_predicate(&pne.symbol),
                pne.args.clone(),
            )));
        }
        rules.push(ExplorationRule {
            conditions,
            effect: Condition::Atom(Atom::new(
                get_action_predicate(&action.exploration_name()),
                action.parameters.iter().map(|p| p.name.clone()).collect(),
            )),
        });

        let action_head = Condition::Atom(Atom::new(
            get_action_predicate(&action.exploration_name()),
            action.parameters.iter().map(|p| p.name.clone()).collect(),
        ));

        for effect in &action.effects {
            if effect.peffect.is_negated() {
                continue;
            }
            let mut conditions = vec![action_head.clone()];
            conditions.extend(condition_to_rule_body(
                &effect.parameters,
                &effect.condition,
            ));
            rules.push(ExplorationRule {
                conditions,
                effect: effect.peffect.clone(),
            });
        }

        for (parameters, condition, assignment) in &action.assign_effects {
            let mut conditions = vec![action_head.clone()];
            conditions.extend(condition_to_rule_body(parameters, condition));

            rules.push(ExplorationRule {
                conditions: conditions.clone(),
                effect: Condition::Atom(Atom::new(
                    get_function_predicate(&assignment.fluent.symbol),
                    assignment.fluent.args.clone(),
                )),
            });

            rules.push(ExplorationRule {
                conditions,
                effect: Condition::Atom(Atom::new(
                    get_fluent_function_predicate(&assignment.fluent.symbol),
                    assignment.fluent.args.clone(),
                )),
            });
        }
    }

    // Axiom applicability and effect rules.
    for axiom in &task.axioms {
        rules.push(ExplorationRule {
            conditions: condition_to_rule_body(&axiom.parameters, &axiom.condition),
            effect: Condition::Atom(Atom::new(
                get_axiom_predicate(&axiom.name),
                axiom.parameters.iter().map(|p| p.name.clone()).collect(),
            )),
        });
        rules.push(ExplorationRule {
            conditions: vec![Condition::Atom(Atom::new(
                get_axiom_predicate(&axiom.name),
                axiom.parameters.iter().map(|p| p.name.clone()).collect(),
            ))],
            effect: Condition::Atom(Atom::new(
                axiom.name.clone(),
                axiom.parameters[..axiom.num_external_parameters]
                    .iter()
                    .map(|p| p.name.clone())
                    .collect(),
            )),
        });
    }

    rules.push(ExplorationRule {
        conditions: condition_to_rule_body(&[], &task.goal),
        effect: Condition::Atom(Atom::new("@goal-reachable".to_string(), vec![])),
    });

    for axiom in task.function_administrator.get_all_axioms() {
        let mut applicability_args: Vec<String> =
            axiom.parameters.iter().map(|p| p.name.clone()).collect();
        for part in &axiom.parts {
            if let FunctionalExpression::PrimitiveNumericExpression(pne) = part {
                applicability_args.extend(pne.args.clone());
            }
        }

        let applicability_head = Condition::Atom(Atom::new(
            get_function_axiom_predicate(&axiom.name),
            applicability_args,
        ));

        let applicability_conditions: Vec<Condition> = axiom
            .parts
            .iter()
            .filter_map(|part| {
                if let FunctionalExpression::PrimitiveNumericExpression(pne) = part {
                    Some(Condition::Atom(Atom::new(
                        get_function_predicate(&pne.symbol),
                        pne.args.clone(),
                    )))
                } else {
                    None
                }
            })
            .collect();
        rules.push(ExplorationRule {
            conditions: applicability_conditions,
            effect: applicability_head.clone(),
        });

        let head = axiom.get_head();
        rules.push(ExplorationRule {
            conditions: vec![applicability_head.clone()],
            effect: Condition::Atom(Atom::new(
                get_function_predicate(&head.symbol),
                head.args.clone(),
            )),
        });

        for part in &axiom.parts {
            if let FunctionalExpression::PrimitiveNumericExpression(pne) = part {
                rules.push(ExplorationRule {
                    conditions: vec![
                        applicability_head.clone(),
                        Condition::Atom(Atom::new(
                            get_fluent_function_predicate(&pne.symbol),
                            pne.args.clone(),
                        )),
                    ],
                    effect: Condition::Atom(Atom::new(
                        get_fluent_function_predicate(&head.symbol),
                        head.args.clone(),
                    )),
                });
            }
        }
    }

    add_derived_expression_demand(&mut rules, task);
    rules
}

/// Demand-driven ("magic sets") exploration of derived numeric expressions.
///
/// Explored bottom-up, a derived expression is instantiated for every tuple
/// of arguments its operands are defined on, whether or not anything reads
/// it. A sum over two parameterized fluents that one action compares then
/// grounds the whole cross product: on pathwaysmetric and the farmland
/// variants that was more than 99.9% of the model, and the preprocessor later
/// discarded all but a few hundred of those derived variables as unnecessary.
///
/// Instead, every rule body that asks whether a derived expression is
/// defined demands it for the arguments it asks about, and the expression's
/// applicability rule waits for that demand. The expressions and actions the
/// task can reach are exactly the same; only the unread instances are gone.
fn add_derived_expression_demand(rules: &mut Vec<ExplorationRule>, task: &Task) {
    let axioms = task.function_administrator.get_all_axioms();
    let parameters_by_defined: HashMap<String, (&str, Vec<String>)> = axioms
        .iter()
        .map(|axiom| {
            (
                get_function_predicate(&axiom.name),
                (
                    axiom.name.as_str(),
                    axiom.parameters.iter().map(|p| p.name.clone()).collect(),
                ),
            )
        })
        .collect();

    // An applicability rule fires only for demanded tuples.
    for rule in rules.iter_mut() {
        let Condition::Atom(head) = &rule.effect else {
            continue;
        };
        let Some(name) = head.predicate.strip_prefix("@function-axiom-") else {
            continue;
        };
        let (_, parameters) = &parameters_by_defined[&get_function_predicate(name)];
        rule.conditions.insert(
            0,
            Condition::Atom(Atom::new(get_demand_predicate(name), parameters.clone())),
        );
    }

    let is_derived = |predicate: &str| parameters_by_defined.contains_key(predicate);
    // `(body, derived expression, arguments)`: a body under which the
    // expression is read for those arguments.
    let mut requests: Vec<(Vec<Condition>, String, Vec<String>)> = Vec::new();

    // Every body that asks whether a derived expression is defined demands
    // it, under the rest of that body.
    for rule in rules.iter() {
        let (asked, rest): (Vec<&Condition>, Vec<&Condition>) = rule.conditions.iter().partition(
            |condition| matches!(condition, Condition::Atom(atom) if is_derived(&atom.predicate)),
        );
        let rest: Vec<Condition> = rest.into_iter().cloned().collect();
        for condition in asked {
            let Condition::Atom(atom) = condition else {
                unreachable!("partitioned on atoms");
            };
            requests.push((rest.clone(), atom.predicate.clone(), atom.args.clone()));
        }
    }

    // Expressions read without a `defined!` guard: the right-hand sides of
    // assignment effects, costs without one, and the metric.
    let mut request_expression = |body: Vec<Condition>, expression: &FunctionalExpression| {
        let body: Vec<Condition> = body
            .into_iter()
            .filter(|condition| {
                !matches!(condition, Condition::Atom(atom) if is_derived(&atom.predicate))
            })
            .collect();
        for pne in expression.primitive_numeric_expressions() {
            let predicate = get_function_predicate(&pne.symbol);
            if is_derived(&predicate) {
                requests.push((body.clone(), predicate, pne.args));
            }
        }
    };
    for action in &task.actions {
        let head = Condition::Atom(Atom::new(
            get_action_predicate(&action.exploration_name()),
            action.parameters.iter().map(|p| p.name.clone()).collect(),
        ));
        for (parameters, condition, assignment) in &action.assign_effects {
            let mut body = vec![head.clone()];
            body.extend(condition_to_rule_body(parameters, condition));
            request_expression(body, &assignment.expression);
        }
        if let Some(cost) = &action.cost {
            let guarded = matches!(&cost.expression,
                FunctionalExpression::PrimitiveNumericExpression(pne) if !pne.args.is_empty());
            if !guarded {
                request_expression(vec![head.clone()], &cost.expression);
            }
        }
    }
    request_expression(
        Vec::new(),
        &FunctionalExpression::PrimitiveNumericExpression(task.metric.1.clone()),
    );

    let mut demand_rules = Vec::new();
    for (body, predicate, args) in requests {
        let (name, _) = parameters_by_defined[&predicate];
        let bound: HashSet<&str> = body
            .iter()
            .flat_map(|condition| match condition {
                Condition::Atom(atom) => atom.args.iter().map(String::as_str),
                other => panic!("exploration rule body holds the non-atom {other}"),
            })
            .collect();
        for argument in args.iter().filter(|arg| arg.starts_with('?')) {
            assert!(
                bound.contains(argument.as_str()),
                "derived expression {name} is read for {argument}, which nothing else in its rule body binds"
            );
        }
        demand_rules.push(ExplorationRule {
            conditions: body,
            effect: Condition::Atom(Atom::new(get_demand_predicate(name), args)),
        });
    }
    rules.extend(demand_rules);
}

/// An exploration rule for the grounding process
#[derive(Debug, Clone)]
pub struct ExplorationRule {
    pub conditions: Vec<Condition>,
    pub effect: Condition,
}

// ==================== Helper predicates ====================

pub fn get_action_predicate(action_name: &str) -> String {
    format!("@action-{}", action_name)
}

pub fn get_axiom_predicate(axiom_name: &str) -> String {
    format!("@axiom-{}", axiom_name)
}

pub fn get_function_predicate(func_name: &str) -> String {
    format!("defined!{}", func_name)
}

pub fn get_fluent_function_predicate(func_name: &str) -> String {
    format!("@fluent-function-{}", func_name)
}

/// The argument tuples of a derived expression some rule body asks about.
pub fn get_demand_predicate(axiom_name: &str) -> String {
    format!("@demand-{}", axiom_name)
}

pub fn get_function_axiom_predicate(axiom_name: &str) -> String {
    format!("@function-axiom-{}", axiom_name)
}

pub fn condition_to_rule_body(parameters: &[TypedObject], condition: &Condition) -> Vec<Condition> {
    let mut result: Vec<Condition> = parameters
        .iter()
        .map(|parameter| Condition::Atom(parameter.get_atom()))
        .collect();

    if matches!(condition, Condition::Truth) {
        return result;
    }

    let mut body_condition = condition.clone();
    if let Condition::ExistentialCondition(ec) = &body_condition {
        for parameter in &ec.parameters {
            result.push(Condition::Atom(parameter.get_atom()));
        }
        if let Some(part) = ec.parts.first() {
            body_condition = part.clone();
        }
    }

    let parts = match body_condition {
        Condition::Conjunction(conj) => conj.parts,
        other => vec![other],
    };

    for part in parts {
        match part {
            Condition::Atom(_) => result.push(part),
            Condition::FunctionComparison(_) | Condition::NegatedFunctionComparison(_) => {
                for pne in part
                    .comparison_operands()
                    .iter()
                    .flat_map(FunctionalExpression::primitive_numeric_expressions)
                {
                    result.push(Condition::Atom(Atom::new(
                        get_function_predicate(&pne.symbol),
                        pne.args,
                    )));
                }
            }
            // A Horn rule body has no negation, and it does not need one: these
            // rules compute an over-approximation of what is reachable, and
            // dropping a negative literal only ever makes that set larger, which
            // is the safe direction. This is the one deliberate omission here.
            Condition::NegatedAtom(_) => {}
            // Truth contributes nothing and Falsity cannot be reached, since
            // instantiation drops a rule whose body is unsatisfiable.
            Condition::Truth | Condition::Falsity => {}
            // A conjunction nested inside a conjunction is flattened rather than
            // rejected. `simplified` removes most of them, but not every path
            // into here has been simplified, and nesting changes nothing about
            // what the rule body means. Panicking on it, as this arm first did,
            // refused tasks that were perfectly well formed.
            Condition::Conjunction(_) => {
                result.extend(condition_to_rule_body(&[], &part));
            }
            // Everything else is gone by the time rules are built: universals
            // became axioms, disjunctions were split, and existentials were
            // moved into the parameter list above. Dropping one silently would
            // widen the rule to fire without its condition, which is how a
            // derived predicate becomes unconditionally true.
            other => panic!(
                "condition {other} should have been normalized away before \
                 exploration rules are built"
            ),
        }
    }

    result
}
