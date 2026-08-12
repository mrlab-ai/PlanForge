/// Normalization of PDDL tasks before grounding.
use std::collections::{HashMap, HashSet};

use tracing::debug;

use super::pddl::axioms::Axiom;
use super::pddl::conditions::*;
use super::pddl::effects::Effect;
use super::pddl::f_expression::*;
use super::pddl::pddl_types::TypedObject;
use super::pddl::tasks::Task;

/// Wraps a Task with additional state for normalization.
pub struct NormalizableTask {
    pub task: Task,
    pub goal: Condition,
}

impl NormalizableTask {
    pub fn from_task(task: Task) -> Self {
        let goal = task.goal.clone();
        NormalizableTask { task, goal }
    }

    pub fn add_global_constraints(&mut self) {
        self.task.add_global_constraints();
    }
}

/// One normalization pass: the name the log calls it by, and the rewrite it
/// applies to the task in place.
type Pass = (&'static str, fn(&mut Task));

/// Every normalization pass, in the order the rest of the pipeline depends on.
/// Quantifiers are gone before the disjunctions are split, and the arithmetic is
/// flattened last, over conditions that are already conjunctions of literals.
///
/// A pass is a plain `fn(&mut Task)`, and adding one is a line here. A trait
/// would buy nothing: there is one implementation of each pass, none is chosen at
/// run time, and a function pointer is already enough to name a pass, list it and
/// trace it. What the list is for is that the pipeline says what it does in one
/// place instead of in a comment per call.
const PASSES: &[Pass] = &[
    ("uniquify variables", uniquify_variables),
    ("convert types to predicates", convert_types_to_predicates),
    ("remove universal quantifiers", remove_universal_quantifiers),
    ("substitute complicated goal", substitute_complicated_goal),
    ("build dnf", build_dnf),
    ("split disjunctions", split_disjunctions),
    ("move existential quantifiers", move_existential_quantifiers),
    (
        "eliminate existential quantifiers",
        eliminate_existential_quantifiers,
    ),
    (
        "remove arithmetic expressions",
        remove_arithmetic_expressions,
    ),
    ("verify axiom predicates", verify_axiom_predicates),
];

/// Runs every pass in [`PASSES`] over the task, then republishes the goal they
/// rewrote. Every pass either succeeds or panics on a task PDDL cannot express,
/// so there is nothing here to report.
pub fn normalize(task: &mut NormalizableTask) {
    for &(name, pass) in PASSES {
        debug!("normalizing: {name}");
        pass(&mut task.task);
    }
    task.goal = task.task.goal.clone();
}

/// Renames every action's and axiom's parameters apart from every other's, so
/// that the passes below may move a condition from one scope into another.
fn uniquify_variables(task: &mut Task) {
    for action in &mut task.actions {
        action.uniquify_variables();
    }
    for axiom in &mut task.axioms {
        axiom.uniquify_variables();
    }
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

/// Every place a task keeps a condition.
///
/// Fast Downward funnels each normalization pass through one generator of these,
/// and that is why its passes cannot miss a site. Here they were written one
/// traversal per pass, each over whichever subset its author had in mind, so a
/// `forall` was expanded in preconditions and a disjunction was split only in
/// actions. Naming the four sites once makes coverage a property of the
/// abstraction instead of a thing to remember.
#[derive(Debug, Clone, Copy)]
enum Site {
    Precondition(usize),
    EffectCondition(usize, usize),
    AxiomBody(usize),
    Goal,
}

/// Every site in the task, as a snapshot.
///
/// A snapshot, not a live iterator, because the passes below append axioms while
/// they run. Fast Downward takes the same snapshot for the same reason: a new
/// axiom's body has already been normalized before it is added, so revisiting it
/// would be wasted work at best and non-terminating at worst.
fn sites(task: &Task) -> Vec<Site> {
    let mut sites = vec![Site::Goal];
    for (a, action) in task.actions.iter().enumerate() {
        sites.push(Site::Precondition(a));
        for e in 0..action.effects.len() {
            sites.push(Site::EffectCondition(a, e));
        }
    }
    sites.extend((0..task.axioms.len()).map(Site::AxiomBody));
    sites
}

fn condition_at(task: &Task, site: Site) -> &Condition {
    match site {
        Site::Precondition(a) => &task.actions[a].precondition,
        Site::EffectCondition(a, e) => &task.actions[a].effects[e].condition,
        Site::AxiomBody(x) => &task.axioms[x].condition,
        Site::Goal => &task.goal,
    }
}

fn set_condition_at(task: &mut Task, site: Site, condition: Condition) {
    match site {
        Site::Precondition(a) => task.actions[a].precondition = condition,
        Site::EffectCondition(a, e) => task.actions[a].effects[e].condition = condition,
        Site::AxiomBody(x) => task.axioms[x].condition = condition,
        Site::Goal => task.goal = condition,
    }
}

/// The parameters in scope at a site, which are what an inner condition's free
/// variables can refer to besides the quantifiers enclosing them.
fn scope_parameters(task: &Task, site: Site) -> Vec<TypedObject> {
    match site {
        Site::Precondition(a) => task.actions[a].parameters.clone(),
        Site::EffectCondition(a, e) => {
            let mut scope = task.actions[a].parameters.clone();
            scope.extend(task.actions[a].effects[e].parameters.clone());
            scope
        }
        Site::AxiomBody(x) => task.axioms[x].parameters.clone(),
        Site::Goal => Vec::new(),
    }
}

/// Rewrites the condition at every site, one for one.
///
/// The rewrite gets the task so it can add axioms, which is what removing a
/// universal needs, and it gets the site's scope so it can type the parameters of
/// the axiom it adds.
fn rewrite_each_site(
    task: &mut Task,
    mut rewrite: impl FnMut(&mut Task, &Condition, &[TypedObject]) -> Condition,
) {
    for site in sites(task) {
        let condition = condition_at(task, site).clone();
        let scope = scope_parameters(task, site);
        let rewritten = rewrite(task, &condition, &scope);
        set_condition_at(task, site, rewritten);
    }
}

/// A variable's declared type, for every variable a site can mention.
///
/// A free variable of a nested condition is bound either by a quantifier above it
/// or by the site's own parameter list, so those two sources together type all of
/// them.
fn type_map(scope: &[TypedObject], condition: &Condition) -> HashMap<String, String> {
    fn collect(condition: &Condition, map: &mut HashMap<String, String>) {
        let bound = match condition {
            Condition::UniversalCondition(UniversalCondition { parameters, .. })
            | Condition::ExistentialCondition(ExistentialCondition { parameters, .. }) => {
                parameters.as_slice()
            }
            _ => &[],
        };
        for parameter in bound {
            map.insert(parameter.name.clone(), parameter.type_name.clone());
        }
        for part in condition.parts() {
            collect(part, map);
        }
    }

    let mut map: HashMap<String, String> = scope
        .iter()
        .map(|parameter| (parameter.name.clone(), parameter.type_name.clone()))
        .collect();
    collect(condition, &mut map);
    map
}

/// Replaces `forall(vars, phi)` by `not new-axiom(free vars)`, where the new
/// axiom derives the negation of the quantified condition.
///
/// This is the one rewrite that cannot be done in place. Grounding enumerates
/// objects for an existential, because it only has to find one witness, and has
/// no way to check a property of every object at once. So a universal is
/// expressed through its dual: `forall(vars, phi)` fails exactly when
/// `exists(vars, not phi)` holds, and that existential is what the new axiom
/// derives. The site then asks for the axiom's atom to be *false*.
///
/// The previous implementation was a no-op that read like the real thing. It
/// matched on the universal, normalized its body, and rebuilt it with
/// `with_parts`, which preserves the variant. Nothing downstream reads a
/// universal, so every `forall` precondition was silently true.
fn remove_universal_quantifiers(task: &mut Task) {
    // One axiom per distinct (condition, parameters) pair, so a condition
    // appearing at several sites shares the axiom it needs rather than adding a
    // copy per site.
    let mut axioms_by_condition: HashMap<(Condition, Vec<TypedObject>), String> = HashMap::new();

    rewrite_each_site(task, |task, condition, scope| {
        if !condition.has_universal_part() {
            return condition.clone();
        }
        let types = type_map(scope, condition);
        recurse_universal(task, condition, &types, &mut axioms_by_condition)
    });
}

fn recurse_universal(
    task: &mut Task,
    condition: &Condition,
    types: &HashMap<String, String>,
    axioms_by_condition: &mut HashMap<(Condition, Vec<TypedObject>), String>,
) -> Condition {
    let Condition::UniversalCondition(_) = condition else {
        let parts = condition
            .parts()
            .iter()
            .map(|part| recurse_universal(task, part, types, axioms_by_condition))
            .collect();
        return condition.with_parts(parts);
    };

    // `not forall(vars, phi)` is `exists(vars, not phi)` in negation normal
    // form, which is what the axiom will derive.
    let axiom_condition = condition.negate();
    let parameters: Vec<TypedObject> = axiom_condition
        .free_variables()
        .into_iter()
        .map(|variable| {
            let type_name = types.get(&variable).unwrap_or_else(|| {
                panic!("variable {variable} is free in {axiom_condition} but has no declared type")
            });
            TypedObject::new(&variable, type_name)
        })
        .collect();

    let key = (axiom_condition.clone(), parameters.clone());
    let name = match axioms_by_condition.get(&key) {
        Some(name) => name.clone(),
        None => {
            // Recurse into the negated body first: it can hold universals of
            // its own, and the axiom has to be added with those already gone.
            let body = recurse_universal(task, &axiom_condition, types, axioms_by_condition);
            // Ask the task for the name rather than counting here: the global
            // constraint invents axioms from the same namespace, and a
            // collision would merge the two bodies into one head.
            let name = task.fresh_axiom_name();
            task.axioms.push(Axiom::new(
                name.clone(),
                parameters.clone(),
                parameters.len(),
                body,
            ));
            axioms_by_condition.insert(key, name.clone());
            name
        }
    };

    Condition::NegatedAtom(NegatedAtom::new(
        name,
        parameters.into_iter().map(|p| p.name).collect(),
    ))
}

/// Hides a goal the SAS+ encoding cannot express behind a derived predicate.
///
/// The encoding takes a conjunction of facts, so a goal whose conjuncts are each
/// [`Condition::is_single_fact`] is emitted as it stands. A numeric comparison is
/// one of those: it gets a condition variable of its own, exactly as it does in a
/// precondition, and the goal names that variable directly. Only a genuinely
/// non-conjunctive goal, disjunctive or quantified or nested, becomes an axiom,
/// and then the goal is the derived atom the axiom proves.
///
/// This runs *before* the disjunctions are split, which is deliberate and is what
/// Fast Downward does. The axiom this creates is itself a site, so the split that
/// follows turns a disjunctive goal into several axioms with one head, which is
/// how the axiom layer spells a disjunction. Splitting first instead would leave
/// the goal a disjunction with nowhere to put it.
fn substitute_complicated_goal(task: &mut Task) {
    let goal = &task.goal;
    let needs_substitution = match goal {
        Condition::Conjunction(conj) => conj.parts.iter().any(|p| !p.is_single_fact()),
        Condition::Truth => false,
        other => !other.is_single_fact(),
    };

    if needs_substitution {
        let new_pred = "@goal-reachable".to_string();
        let axiom = Axiom::new(new_pred.clone(), vec![], 0, goal.clone());
        task.axioms.push(axiom);
        task.goal = Condition::Atom(Atom::new(new_pred, vec![]));
    }
}

/// Pulls every disjunction to the root of the condition it appears in.
///
/// Once universals are gone, three rewrites suffice, and they are the three Fast
/// Downward names:
///
/// 1. `or(phi, or(psi, chi))` is `or(phi, psi, chi)`.
/// 2. `exists(vars, or(phi, psi))` is `or(exists(vars, phi), exists(vars, psi))`.
/// 3. `and(phi, or(psi, chi))` is `or(and(phi, psi), and(phi, chi))`.
///
/// Rule 2 is the one the previous code was missing entirely, along with running
/// over anything but action preconditions. Without it an `or` nested inside an
/// `exists` never reaches the root and so is never split.
fn build_dnf(task: &mut Task) {
    rewrite_each_site(task, |_task, condition, _scope| {
        if !condition.has_disjunction() {
            return condition.clone();
        }
        to_dnf(condition).simplified()
    });
}

fn to_dnf(condition: &Condition) -> Condition {
    let mut disjunctions: Vec<Disjunction> = Vec::new();
    let mut others: Vec<Condition> = Vec::new();
    for part in condition.parts() {
        match to_dnf(part) {
            Condition::Disjunction(disjunction) => disjunctions.push(disjunction),
            other => others.push(other),
        }
    }
    if disjunctions.is_empty() {
        return condition.with_parts(others);
    }

    match condition {
        // Rule 1: associativity of disjunction.
        Condition::Disjunction(_) => {
            let mut parts = others;
            for disjunction in disjunctions {
                parts.extend(disjunction.parts);
            }
            Condition::Disjunction(Disjunction::new(parts))
        }
        // Rule 2: an existential distributes over a disjunction. After the
        // recursion above a quantifier holds exactly one part, so there is
        // exactly one disjunction to distribute over.
        Condition::ExistentialCondition(existential) => {
            let inner = disjunctions
                .into_iter()
                .next()
                .expect("checked non-empty above");
            Condition::Disjunction(Disjunction::new(
                inner
                    .parts
                    .into_iter()
                    .map(|part| {
                        Condition::ExistentialCondition(ExistentialCondition::new(
                            existential.parameters.clone(),
                            vec![part],
                        ))
                    })
                    .collect(),
            ))
        }
        // Rule 3: a conjunction distributes over its disjunctions.
        Condition::Conjunction(_) => {
            let mut products = vec![Condition::Conjunction(Conjunction::new(others))];
            for disjunction in disjunctions {
                let mut next = Vec::with_capacity(products.len() * disjunction.parts.len());
                for left in &products {
                    for right in &disjunction.parts {
                        next.push(Condition::Conjunction(Conjunction::new(vec![
                            left.clone(),
                            right.clone(),
                        ])));
                    }
                }
                products = next;
            }
            Condition::Disjunction(Disjunction::new(products))
        }
        // A universal is gone by now, and a leaf has no parts to have produced a
        // disjunction, so nothing else can hold one.
        other => unreachable!("{other} cannot contain a disjunction at this point"),
    }
}

/// Splits every site whose condition is a disjunction into one copy per disjunct.
///
/// What a copy means differs per site, which is the whole reason the sites are
/// named. An action becomes several actions. A conditional effect becomes several
/// effects on the same action. An axiom becomes several axioms *with the same
/// head*, and that is what makes a disjunctive `:derived` body work at all:
/// several bodies proving one head already act as a disjunction, so the axiom
/// layer needs no notion of `or`.
///
/// The previous implementation split actions only, which is why an `or` in a
/// `:derived` body, and the axiom that a disjunctive goal is compiled into, both
/// survived to grounding and were read as unconditionally true.
fn split_disjunctions(task: &mut Task) {
    // Every axiom, including the one substitute_complicated_goal just added for
    // a disjunctive goal.
    let mut axioms = Vec::with_capacity(task.axioms.len());
    for axiom in std::mem::take(&mut task.axioms) {
        match &axiom.condition {
            Condition::Disjunction(disjunction) => {
                for part in disjunction.parts.clone() {
                    let mut copy = axiom.clone();
                    copy.condition = part;
                    axioms.push(copy);
                }
            }
            _ => axioms.push(axiom),
        }
    }
    task.axioms = axioms;

    let mut actions = Vec::with_capacity(task.actions.len());
    for mut action in std::mem::take(&mut task.actions) {
        let mut effects = Vec::with_capacity(action.effects.len());
        for effect in &action.effects {
            match &effect.condition {
                Condition::Disjunction(disjunction) => {
                    for part in &disjunction.parts {
                        effects.push(Effect::new(
                            effect.parameters.clone(),
                            part.clone(),
                            effect.peffect.clone(),
                        ));
                    }
                }
                _ => effects.push(effect.clone()),
            }
        }
        action.effects = effects;

        match &action.precondition {
            Condition::Disjunction(disjunction) => {
                // The copies need distinct names: an action's name identifies it
                // downstream, in the exploration rules and in the plan file.
                for (index, part) in disjunction.parts.iter().enumerate() {
                    let mut copy = action.clone();
                    copy.name = format!("{}@split{index}", action.name);
                    copy.precondition = part.clone();
                    actions.push(copy);
                }
            }
            _ => actions.push(action),
        }
    }
    task.actions = actions;

    // The goal is not a site that can be duplicated: there is one goal. A
    // disjunctive one is hidden behind an axiom by the previous pass, so a
    // disjunction surviving here means that pass failed to fire.
    assert!(
        !matches!(task.goal, Condition::Disjunction(_)),
        "a disjunctive goal must be compiled into an axiom before disjunctions are split"
    );
}

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
                        vec![
                            Condition::Conjunction(Conjunction::new(new_conjunction_parts))
                                .simplified(),
                        ],
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
            Condition::Disjunction(_) | Condition::UniversalCondition(_) => {
                condition.map_parts(recurse).simplified()
            }
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
            action.parameters.extend(ec.parameters.clone());
            action.precondition = existential_body(ec);
        }
    }
}

fn eliminate_existential_quantifiers_from_conditional_effects(task: &mut Task) {
    for action in &mut task.actions {
        for eff in &mut action.effects {
            if let Condition::ExistentialCondition(ec) = &eff.condition {
                eff.parameters.extend(ec.parameters.clone());
                eff.condition = existential_body(ec);
            }
        }
    }
}

fn eliminate_existential_quantifiers_from_axioms(task: &mut Task) {
    for axiom in &mut task.axioms {
        if let Condition::ExistentialCondition(ec) = &axiom.condition {
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

/// Creates numeric axioms for complex arithmetic expressions.
fn remove_arithmetic_expressions(task: &mut Task) {
    fn rewrite_condition(
        function_administrator: &mut super::pddl::tasks::DerivedFunctionAdministrator,
        condition: &Condition,
    ) -> Condition {
        match condition {
            Condition::FunctionComparison(_) | Condition::NegatedFunctionComparison(_) => condition
                .map_comparison_operands(|operand| {
                    FunctionalExpression::PrimitiveNumericExpression(
                        function_administrator.get_derived_function(operand),
                    )
                }),
            other => other.map_parts(|part| rewrite_condition(function_administrator, part)),
        }
    }

    for action in &mut task.actions {
        let precondition = action.precondition.clone();
        action.precondition = rewrite_condition(&mut task.function_administrator, &precondition);
        for eff in &mut action.effects {
            let condition = eff.condition.clone();
            eff.condition = rewrite_condition(&mut task.function_administrator, &condition);
            if let Condition::FunctionComparison(_) | Condition::NegatedFunctionComparison(_) =
                &eff.peffect
            {
                let peffect = eff.peffect.clone();
                eff.peffect = rewrite_condition(&mut task.function_administrator, &peffect);
            }
        }
        for (_, _, assignment) in &mut action.assign_effects {
            if !matches!(
                assignment.expression,
                FunctionalExpression::PrimitiveNumericExpression(_)
            ) {
                let expression = assignment.expression.clone();
                assignment.expression = FunctionalExpression::PrimitiveNumericExpression(
                    task.function_administrator
                        .get_derived_function(&expression),
                );
            }
        }
        if let Some(cost) = &mut action.cost
            && !matches!(
                cost.expression,
                FunctionalExpression::PrimitiveNumericExpression(_)
            )
        {
            let expression = cost.expression.clone();
            cost.expression = FunctionalExpression::PrimitiveNumericExpression(
                task.function_administrator
                    .get_derived_function(&expression),
            );
        }
    }

    for axiom in &mut task.axioms {
        let condition = axiom.condition.clone();
        axiom.condition = rewrite_condition(&mut task.function_administrator, &condition);
    }

    let goal = task.goal.clone();
    task.goal = rewrite_condition(&mut task.function_administrator, &goal);
}

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
            if let Some(pred) = eff.peffect.literal_predicate()
                && axiom_names.contains(pred)
            {
                panic!(
                    "error: derived predicate {:?} appears in effect of action {:?}",
                    pred, action.name
                );
            }
        }
    }
}

// ==================== Exploration rules ====================

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

    rules
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
