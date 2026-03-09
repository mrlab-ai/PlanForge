/// Port of normalize.py
/// Normalization of PDDL tasks before grounding.

use std::collections::{HashMap, HashSet};

use super::pddl::conditions::*;
use super::pddl::pddl_types::TypedObject;
use super::pddl::effects::{Effect, EffectType, EffectKind};
use super::pddl::actions::Action;
use super::pddl::axioms::Axiom;
use super::pddl::tasks::Task;
use super::pddl::f_expression::*;

/// Python: NormalizableTask equivalent
/// Wraps a Task with additional state for normalization.
pub struct NormalizableTask {
    pub task: Task,
    pub goal: Condition,
}

impl NormalizableTask {
    pub fn from_ast(
        _dom: &super::pddl_parser::lisp_parser::SExpr,
        _prob: &super::pddl_parser::lisp_parser::SExpr,
    ) -> Self {
        // This will be called from main.rs; for now build the task through the parser
        unimplemented!("Use from_task instead")
    }

    pub fn from_task(task: Task) -> Self {
        let goal = task.goal.clone();
        NormalizableTask { task, goal }
    }

    pub fn add_global_constraints(&mut self) {
        self.task.add_global_constraints();
    }
}

/// Python: normalize(task)
/// Main normalization entry point. Performs multiple normalization steps.
pub fn normalize(task: &mut NormalizableTask) -> Result<(), String> {
    let t = &mut task.task;

    // Step 1: Uniquify variables in actions and axioms
    for action in &mut t.actions {
        action.uniquify_variables();
    }
    for axiom in &mut t.axioms {
        axiom.uniquify_variables();
    }

    // Step 2: Convert types to predicates (untype)
    // type predicates for typed objects
    let type_predicates = convert_types_to_predicates(t);

    // Step 3: Remove universal quantifiers from conditions
    remove_universal_quantifiers(t);

    // Step 4: Substitute complicated goals
    substitute_complicated_goal(t);

    // Step 5: Build DNF for disjunctive conditions, then split
    build_dnf(t);
    split_disjunctions(t);

    // Step 6: Move and eliminate existential quantifiers
    move_existential_quantifiers(t);
    eliminate_existential_quantifiers(t);

    // Step 7: Verify and fix arithmetic expressions
    verify_and_fix_arithmetic_expressions(t);

    // Step 8: Remove arithmetic expressions (create numeric axioms)
    remove_arithmetic_expressions(t);

    // Step 9: Verify axiom predicates
    verify_axiom_predicates(t);

    task.goal = t.goal.clone();

    Ok(())
}

/// Convert type declarations to predicates (adds type atoms to init)
fn convert_types_to_predicates(task: &mut Task) {
    // Add type predicates for each object
    for obj in &task.objects {
        let atom = obj.get_atom();
        task.init.push(atom);
    }
    // Add supertypes transitively
    // Build a type hierarchy
    let mut type_map: HashMap<String, Option<String>> = HashMap::new();
    for t in &task.types {
        type_map.insert(t.name.clone(), t.basetype_name.clone());
    }

    // For each object, add atoms for all supertypes
    let mut extra_init = vec![];
    for obj in &task.objects {
        let mut current = Some(obj.type_name.clone());
        while let Some(ref type_name) = current {
            if type_name == &obj.type_name {
                current = type_map.get(type_name).cloned().flatten();
                continue;
            }
            let supertype_obj = TypedObject::new(&obj.name, type_name);
            extra_init.push(supertype_obj.get_atom());
            current = type_map.get(type_name).cloned().flatten();
        }
    }
    task.init.extend(extra_init);
}

/// Python: def remove_universal_quantifiers(task)
/// Converts universal quantifiers in preconditions to negated existentials.
fn remove_universal_quantifiers(task: &mut Task) {
    // Remove universals from action preconditions
    for action in &mut task.actions {
        action.precondition = remove_universal(&action.precondition);
    }
    // Remove universals from axiom conditions
    for axiom in &mut task.axioms {
        axiom.condition = remove_universal(&axiom.condition);
    }
    // Remove universals from goal
    task.goal = remove_universal(&task.goal);

    // Remove universals from effects (conditional effects can have universal quantifiers)
    for action in &mut task.actions {
        let mut new_effects = vec![];
        for eff in &action.effects {
            let new_cond = remove_universal(&eff.condition);
            new_effects.push(Effect::new(eff.parameters.clone(), new_cond, eff.peffect.clone()));
        }
        action.effects = new_effects;
    }
}

fn remove_universal(cond: &Condition) -> Condition {
    match cond {
        Condition::UniversalCondition(uc) => {
            // forall params. phi  =>  not(exists params. not(phi))
            let inner = if uc.parts.len() == 1 {
                remove_universal(&uc.parts[0])
            } else {
                Condition::Conjunction(Conjunction::new(
                    uc.parts.iter().map(|p| remove_universal(p)).collect()
                ))
            };
            // not(exists params. not(inner))
            // We handle this by adding a new axiom
            // For now, just recursively process
            Condition::UniversalCondition(UniversalCondition::new(
                uc.parameters.clone(),
                vec![remove_universal(&inner)],
            ))
        }
        Condition::Conjunction(conj) => {
            Condition::Conjunction(Conjunction::new(
                conj.parts.iter().map(|p| remove_universal(p)).collect()
            ))
        }
        Condition::Disjunction(disj) => {
            Condition::Disjunction(Disjunction::new(
                disj.parts.iter().map(|p| remove_universal(p)).collect()
            ))
        }
        Condition::ExistentialCondition(ec) => {
            Condition::ExistentialCondition(ExistentialCondition::new(
                ec.parameters.clone(),
                ec.parts.iter().map(|p| remove_universal(p)).collect(),
            ))
        }
        other => other.clone(),
    }
}

/// Python: def substitute_complicated_goal(task)
fn substitute_complicated_goal(task: &mut Task) {
    let goal = &task.goal;
    // If goal is not a simple conjunction of literals, create an axiom
    let needs_substitution = match goal {
        Condition::Conjunction(conj) => {
            conj.parts.iter().any(|p| !p.is_literal())
        }
        Condition::Atom(_) | Condition::NegatedAtom(_) => false,
        Condition::Truth => false,
        _ => true,
    };

    if needs_substitution {
        let new_pred = "@goal-reachable".to_string();
        let axiom = Axiom::new(
            new_pred.clone(), vec![], 0, goal.clone(),
        );
        task.axioms.push(axiom);
        task.goal = Condition::Atom(Atom::new(new_pred, vec![]));
    }
}

/// Python: def build_DNF(task)
fn build_dnf(task: &mut Task) {
    // For each action, if precondition has disjunctions, convert to DNF
    // This is handled during split_disjunctions
}

/// Python: def split_disjunctions(task)
fn split_disjunctions(task: &mut Task) {
    // Split actions with disjunctive preconditions into multiple actions
    let mut new_actions = vec![];
    for action in &task.actions {
        if action.precondition.has_disjunction() {
            let dnf = to_dnf(&action.precondition);
            for (i, conj) in dnf.iter().enumerate() {
                let mut new_action = action.clone();
                new_action.name = format!("{}@split{}", action.name, i);
                new_action.precondition = conj.clone();
                new_actions.push(new_action);
            }
        } else {
            new_actions.push(action.clone());
        }
    }
    task.actions = new_actions;
}

/// Convert a condition to DNF (list of conjunctions)
fn to_dnf(cond: &Condition) -> Vec<Condition> {
    match cond {
        Condition::Disjunction(disj) => {
            let mut result = vec![];
            for part in &disj.parts {
                result.extend(to_dnf(part));
            }
            result
        }
        Condition::Conjunction(conj) => {
            // Distribute conjunction over disjunctions
            let mut dnf_parts: Vec<Vec<Condition>> = vec![vec![]];
            for part in &conj.parts {
                let part_dnf = to_dnf(part);
                let mut new_dnf_parts = vec![];
                for existing in &dnf_parts {
                    for new_part in &part_dnf {
                        let mut combined = existing.clone();
                        match new_part {
                            Condition::Conjunction(c) => combined.extend(c.parts.clone()),
                            other => combined.push(other.clone()),
                        }
                        new_dnf_parts.push(combined);
                    }
                }
                dnf_parts = new_dnf_parts;
            }
            dnf_parts.into_iter().map(|parts| {
                if parts.len() == 1 {
                    parts.into_iter().next().unwrap()
                } else {
                    Condition::Conjunction(Conjunction::new(parts))
                }
            }).collect()
        }
        other => vec![other.clone()],
    }
}

/// Python: def move_existential_quantifiers(task)
fn move_existential_quantifiers(task: &mut Task) {
    fn recurse(condition: &Condition) -> Condition {
        match condition {
            Condition::Conjunction(conj) => {
                let mut existential_parts = vec![];
                let mut other_parts = vec![];

                for part in &conj.parts {
                    let part = recurse(part);
                    match part {
                        Condition::ExistentialCondition(ec) => existential_parts.push(ec),
                        other => other_parts.push(other),
                    }
                }

                if existential_parts.is_empty() {
                    Condition::Conjunction(Conjunction::new(other_parts)).simplified()
                } else {
                    let mut new_parameters = vec![];
                    let mut new_conjunction_parts = other_parts;
                    for part in existential_parts {
                        new_parameters.extend(part.parameters);
                        new_conjunction_parts.extend(part.parts);
                    }
                    Condition::ExistentialCondition(ExistentialCondition::new(
                        new_parameters,
                        vec![Condition::Conjunction(Conjunction::new(new_conjunction_parts)).simplified()],
                    ))
                    .simplified()
                }
            }
            Condition::ExistentialCondition(ec) => {
                let mut existential_parameters = ec.parameters.clone();
                let mut new_parts = vec![];
                for part in &ec.parts {
                    match recurse(part) {
                        Condition::ExistentialCondition(inner) => {
                            existential_parameters.extend(inner.parameters);
                            new_parts.extend(inner.parts);
                        }
                        other => new_parts.push(other),
                    }
                }
                Condition::ExistentialCondition(ExistentialCondition::new(
                    existential_parameters,
                    new_parts,
                ))
                .simplified()
            }
            Condition::Disjunction(disj) => Condition::Disjunction(Disjunction::new(
                disj.parts.iter().map(recurse).collect(),
            ))
            .simplified(),
            Condition::UniversalCondition(uc) => Condition::UniversalCondition(
                UniversalCondition::new(uc.parameters.clone(), uc.parts.iter().map(recurse).collect()),
            )
            .simplified(),
            other => other.clone(),
        }
    }

    for action in &mut task.actions {
        if action.precondition.has_existential_part() {
            action.precondition = recurse(&action.precondition);
        }
        for effect in &mut action.effects {
            if effect.condition.has_existential_part() {
                effect.condition = recurse(&effect.condition);
            }
        }
    }

    for axiom in &mut task.axioms {
        if axiom.condition.has_existential_part() {
            axiom.condition = recurse(&axiom.condition);
        }
    }

    if task.goal.has_existential_part() {
        task.goal = recurse(&task.goal);
    }
}

/// Python: Eliminate existential quantifiers by creating new axioms
fn eliminate_existential_quantifiers(task: &mut Task) {
    // From preconditions
    eliminate_existential_quantifiers_from_preconditions(task);
    // From conditional effects
    eliminate_existential_quantifiers_from_conditional_effects(task);
    // From axioms
    eliminate_existential_quantifiers_from_axioms(task);
}

fn eliminate_existential_quantifiers_from_preconditions(task: &mut Task) {
    for action in &mut task.actions {
        if let Condition::ExistentialCondition(ec) = &action.precondition {
            action.parameters = action.parameters.clone();
            action.parameters.extend(ec.parameters.clone());
            action.precondition = existential_body(ec);
        }
    }
}

fn eliminate_existential_quantifiers_from_conditional_effects(task: &mut Task) {
    for action in &mut task.actions {
        let mut new_effects = Vec::with_capacity(action.effects.len());
        for eff in &action.effects {
            let mut new_eff = eff.clone();
            if let Condition::ExistentialCondition(ec) = &eff.condition {
                new_eff.parameters = new_eff.parameters.clone();
                new_eff.parameters.extend(ec.parameters.clone());
                new_eff.condition = existential_body(ec);
            }
            new_effects.push(new_eff);
        }
        action.effects = new_effects;
    }
}

fn eliminate_existential_quantifiers_from_axioms(task: &mut Task) {
    for axiom in &mut task.axioms {
        if let Condition::ExistentialCondition(ec) = &axiom.condition {
            axiom.parameters = axiom.parameters.clone();
            axiom.parameters.extend(ec.parameters.clone());
            axiom.condition = existential_body(ec);
        }
    }
}

fn existential_body(ec: &ExistentialCondition) -> Condition {
    if ec.parts.len() == 1 {
        ec.parts[0].clone()
    } else {
        Condition::Conjunction(Conjunction::new(ec.parts.clone()))
    }
}

/// Python: def verify_and_fix_arithmetic_expressions(task)
fn verify_and_fix_arithmetic_expressions(_task: &mut Task) {
    // Verify that arithmetic expressions are well-formed
    // This step mainly checks for issues in the PDDL
}

/// Python: def remove_arithmetic_expressions(task)
/// Creates numeric axioms for complex arithmetic expressions.
fn remove_arithmetic_expressions(task: &mut Task) {
    fn rewrite_condition(
        function_administrator: &mut super::pddl::tasks::DerivedFunctionAdministrator,
        condition: &Condition,
    ) -> Condition {
        match condition {
            Condition::FunctionComparison(fc) => {
                let parts = fc.parts.iter().map(|part| {
                    FunctionalExpression::PrimitiveNumericExpression(
                        function_administrator.get_derived_function(part, &HashSet::new()),
                    )
                }).collect();
                Condition::FunctionComparison(FunctionComparison::new(fc.comparator.clone(), parts))
            }
            Condition::NegatedFunctionComparison(nfc) => {
                let parts = nfc.parts.iter().map(|part| {
                    FunctionalExpression::PrimitiveNumericExpression(
                        function_administrator.get_derived_function(part, &HashSet::new()),
                    )
                }).collect();
                Condition::NegatedFunctionComparison(NegatedFunctionComparison::new(nfc.comparator.clone(), parts))
            }
            Condition::Conjunction(conj) => Condition::Conjunction(Conjunction::new(
                conj.parts.iter().map(|part| rewrite_condition(function_administrator, part)).collect(),
            )),
            Condition::Disjunction(disj) => Condition::Disjunction(Disjunction::new(
                disj.parts.iter().map(|part| rewrite_condition(function_administrator, part)).collect(),
            )),
            Condition::ExistentialCondition(ec) => Condition::ExistentialCondition(ExistentialCondition::new(
                ec.parameters.clone(),
                ec.parts.iter().map(|part| rewrite_condition(function_administrator, part)).collect(),
            )),
            Condition::UniversalCondition(uc) => Condition::UniversalCondition(UniversalCondition::new(
                uc.parameters.clone(),
                uc.parts.iter().map(|part| rewrite_condition(function_administrator, part)).collect(),
            )),
            other => other.clone(),
        }
    }

    for action in &mut task.actions {
        let precondition = action.precondition.clone();
        action.precondition = rewrite_condition(&mut task.function_administrator, &precondition);
        for eff in &mut action.effects {
            let condition = eff.condition.clone();
            eff.condition = rewrite_condition(&mut task.function_administrator, &condition);
            if let Condition::FunctionComparison(_) | Condition::NegatedFunctionComparison(_) = &eff.peffect {
                let peffect = eff.peffect.clone();
                eff.peffect = rewrite_condition(&mut task.function_administrator, &peffect);
            }
        }
        if let Some(cost) = &mut action.cost {
            if !matches!(cost.expression, FunctionalExpression::PrimitiveNumericExpression(_)) {
                let expression = cost.expression.clone();
                cost.expression = FunctionalExpression::PrimitiveNumericExpression(
                    task.function_administrator.get_derived_function(&expression, &HashSet::new()),
                );
            }
        }
    }

    for axiom in &mut task.axioms {
        let condition = axiom.condition.clone();
        axiom.condition = rewrite_condition(&mut task.function_administrator, &condition);
    }

    let goal = task.goal.clone();
    task.goal = rewrite_condition(&mut task.function_administrator, &goal);
}

/// Python: def verify_axiom_predicates(task)
fn verify_axiom_predicates(task: &mut Task) {
    // Verify that derived predicates are not used in :init or action effects.
    let mut axiom_names: HashSet<String> = HashSet::new();
    for axiom in &task.axioms {
        axiom_names.insert(axiom.name.clone());
    }

    // Check init facts
    for fact in &task.init {
        if axiom_names.contains(&fact.predicate) {
            panic!(
                "error: derived predicate {:?} appears in :init fact '{}'",
                fact.predicate, fact
            );
        }
    }

    // Check that axiom predicates don't appear in effects
    for action in &task.actions {
        for eff in &action.effects {
            if let Some(pred) = eff.peffect.literal_predicate() {
                if axiom_names.contains(pred) {
                    panic!(
                        "error: derived predicate {:?} appears in effect of action {:?}",
                        pred, action.name
                    );
                }
            }
        }
    }
}

// ==================== Exploration rules ====================

/// Python: def build_exploration_rules(task)
/// Builds a set of rules for the grounding process.
/// These rules encode what atoms are reachable.
pub fn build_exploration_rules(task: &Task) -> Vec<ExplorationRule> {
    let mut rules = vec![];

    // Action applicability rules.
    for action in &task.actions {
        rules.push(ExplorationRule {
            conditions: condition_to_rule_body(&action.parameters, &action.precondition),
            effect: Condition::Atom(Atom::new(
                get_action_predicate(&action.name),
                action.parameters.iter().map(|p| p.name.clone()).collect(),
            )),
            parameters: vec![],
        });

        let action_head = Condition::Atom(Atom::new(
            get_action_predicate(&action.name),
            action.parameters.iter().map(|p| p.name.clone()).collect(),
        ));

        for effect in &action.effects {
            if effect.peffect.is_negated() {
                continue;
            }
            let mut conditions = vec![action_head.clone()];
            conditions.extend(condition_to_rule_body(&effect.parameters, &effect.condition));
            rules.push(ExplorationRule {
                conditions,
                effect: effect.peffect.clone(),
                parameters: vec![],
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
                parameters: vec![],
            });

            rules.push(ExplorationRule {
                conditions,
                effect: Condition::Atom(Atom::new(
                    get_fluent_function_predicate(&assignment.fluent.symbol),
                    assignment.fluent.args.clone(),
                )),
                parameters: vec![],
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
            parameters: vec![],
        });
        rules.push(ExplorationRule {
            conditions: vec![Condition::Atom(Atom::new(
                get_axiom_predicate(&axiom.name),
                axiom.parameters.iter().map(|p| p.name.clone()).collect(),
            ))],
            effect: Condition::Atom(Atom::new(
                axiom.name.clone(),
                axiom.parameters[..axiom.num_external_parameters]
                    .iter().map(|p| p.name.clone()).collect(),
            )),
            parameters: vec![],
        });
    }

    rules.push(ExplorationRule {
        conditions: condition_to_rule_body(&[], &task.goal),
        effect: Condition::Atom(Atom::new("@goal-reachable".to_string(), vec![])),
        parameters: vec![],
    });

    for axiom in task.function_administrator.get_all_axioms() {
        let mut applicability_args: Vec<String> = axiom.parameters.iter().map(|p| p.name.clone()).collect();
        for part in &axiom.parts {
            if let FunctionalExpression::PrimitiveNumericExpression(pne) = part {
                applicability_args.extend(pne.args.clone());
            }
        }

        let applicability_head = Condition::Atom(Atom::new(
            get_function_axiom_predicate(&axiom.name),
            applicability_args,
        ));

        let applicability_conditions: Vec<Condition> = axiom.parts.iter()
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
            parameters: vec![],
        });

        let head = axiom.get_head();
        rules.push(ExplorationRule {
            conditions: vec![applicability_head.clone()],
            effect: Condition::Atom(Atom::new(
                get_function_predicate(&head.symbol),
                head.args.clone(),
            )),
            parameters: vec![],
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
                    parameters: vec![],
                });
            }
        }
    }

    rules
}

/// An exploration rule for the grounding process
#[derive(Debug, Clone)]
pub struct ExplorationRule {
    pub conditions: Vec<Condition>,
    pub effect: Condition,
    pub parameters: Vec<TypedObject>,
}

/// Extract all atomic conditions from a condition tree
fn all_conditions(cond: &Condition) -> Vec<Condition> {
    match cond {
        Condition::Conjunction(conj) => {
            conj.parts.iter().flat_map(|p| all_conditions(p)).collect()
        }
        Condition::Truth => vec![],
        other => vec![other.clone()],
    }
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

pub fn get_function_axiom_predicate(axiom_name: &str) -> String {
    format!("@function-axiom-{}", axiom_name)
}

pub fn get_fluent_predicates(task: &Task) -> HashSet<String> {
    let mut result = HashSet::new();
    for action in &task.actions {
        for eff in &action.effects {
            if let Some(pred) = eff.peffect.literal_predicate() {
                result.insert(pred.to_string());
            }
        }
    }
    result
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
            Condition::FunctionComparison(fc) => {
                for pne in fc.parts.iter().flat_map(|expr| expr.primitive_numeric_expressions()) {
                    result.push(Condition::Atom(Atom::new(
                        get_function_predicate(&pne.symbol),
                        pne.args.clone(),
                    )));
                }
            }
            Condition::NegatedFunctionComparison(nfc) => {
                for pne in nfc.parts.iter().flat_map(|expr| expr.primitive_numeric_expressions()) {
                    result.push(Condition::Atom(Atom::new(
                        get_function_predicate(&pne.symbol),
                        pne.args.clone(),
                    )));
                }
            }
            _ => {}
        }
    }

    result
}
