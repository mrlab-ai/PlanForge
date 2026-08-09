/// Port of instantiate.py
/// Instantiates the PDDL task using the logic program model.
use std::collections::{HashMap, HashSet};

use super::build_model;
use super::pddl::actions::PropositionalAction;
use super::pddl::axioms::{InstantiatedNumericAxiom, PropositionalAxiom};
use super::pddl::conditions::*;
use super::pddl::f_expression::*;
use super::pddl::pddl_types::TypedObject;
use super::pddl::tasks::Task;
use super::pddl_to_prolog;

fn collect_used_derived_pnes_from_expr(
    expr: &FunctionalExpression,
    out: &mut HashSet<PrimitiveNumericExpression>,
) {
    match expr {
        FunctionalExpression::PrimitiveNumericExpression(pne) => {
            if pne.symbol.starts_with("derived!") {
                out.insert(pne.clone());
            }
        }
        FunctionalExpression::ArithmeticExpression(ae) => {
            for part in &ae.parts {
                collect_used_derived_pnes_from_expr(part, out);
            }
        }
        FunctionalExpression::AdditiveInverse(ai) => {
            for part in &ai.parts {
                collect_used_derived_pnes_from_expr(part, out);
            }
        }
        FunctionalExpression::NumericConstant(_) => {}
    }
}

fn collect_used_derived_pnes_from_condition(
    cond: &Condition,
    out: &mut HashSet<PrimitiveNumericExpression>,
) {
    match cond {
        Condition::FunctionComparison(fc) => {
            for part in &fc.parts {
                collect_used_derived_pnes_from_expr(part, out);
            }
        }
        Condition::NegatedFunctionComparison(nfc) => {
            for part in &nfc.parts {
                collect_used_derived_pnes_from_expr(part, out);
            }
        }
        Condition::Conjunction(conj) => {
            for part in &conj.parts {
                collect_used_derived_pnes_from_condition(part, out);
            }
        }
        Condition::Disjunction(disj) => {
            for part in &disj.parts {
                collect_used_derived_pnes_from_condition(part, out);
            }
        }
        Condition::ExistentialCondition(ec) => {
            for part in &ec.parts {
                collect_used_derived_pnes_from_condition(part, out);
            }
        }
        Condition::UniversalCondition(uc) => {
            for part in &uc.parts {
                collect_used_derived_pnes_from_condition(part, out);
            }
        }
        _ => {}
    }
}

/// Result of the exploration/instantiation process.
pub struct ExploreResult {
    pub relaxed_reachable: bool,
    pub atoms: Vec<Atom>,
    pub num_fluents: Vec<PrimitiveNumericExpression>,
    pub grounded_ops: Vec<PropositionalAction>,
    pub grounded_axioms: Vec<PropositionalAxiom>,
    pub numeric_axioms: Vec<InstantiatedNumericAxiom>,
    pub init_constant_predicates: Vec<Atom>,
    pub init_constant_numerics: Vec<FunctionAssignment>,
    pub reachable_action_params: HashMap<String, Vec<Vec<String>>>,
}

/// A predicate is fluent if it appears in some action effect or is derived.
fn get_fluent_facts(task: &Task) -> HashSet<String> {
    let mut fluent_preds: HashSet<String> = HashSet::new();
    for action in &task.actions {
        for eff in &action.effects {
            if let Some(pred) = eff.peffect.literal_predicate() {
                fluent_preds.insert(pred.to_string());
            }
        }
    }
    for axiom in &task.axioms {
        fluent_preds.insert(axiom.name.clone());
    }
    fluent_preds
}

/// What a grounded predicate contributes to the instantiated task. Grounding
/// derives one atom per reachable fact, so the classification is done once per
/// predicate rather than by re-testing every atom's name.
struct Role<'a> {
    /// The predicate is fluent, so its atoms become SAS facts. Independent of
    /// `bookkeeping`: normalization marks `@goal-reachable` fluent as well.
    fluent: bool,
    bookkeeping: Bookkeeping<'a>,
}

/// The roles normalization gives to the predicates it introduces itself.
enum Bookkeeping<'a> {
    None,
    /// A reachable numeric fluent, named by its function symbol.
    Function(&'a str),
    /// A reachable parameter tuple of the named action.
    ActionParameters(&'a str),
    /// A reachable instance of the named numeric axiom.
    FunctionAxiom(&'a str),
    /// The goal is reachable in the delete relaxation.
    GoalReachable,
}

impl<'a> Role<'a> {
    fn classify(predicate: &'a str, fluent_facts: &HashSet<String>) -> Self {
        let bookkeeping = if let Some(symbol) = predicate.strip_prefix("@fluent-function-") {
            Bookkeeping::Function(symbol)
        } else if let Some(name) = predicate.strip_prefix("@action-") {
            Bookkeeping::ActionParameters(name)
        } else if let Some(name) = predicate.strip_prefix("@function-axiom-") {
            Bookkeeping::FunctionAxiom(name)
        } else if predicate == "@goal-reachable" {
            Bookkeeping::GoalReachable
        } else {
            Bookkeeping::None
        };
        Role {
            fluent: fluent_facts.contains(predicate),
            bookkeeping,
        }
    }
}

/// Numeric fluents are typed by their symbol: the metric is integral, a
/// function introduced by normalization is derived, everything else is real.
fn function_type(symbol: &str) -> char {
    match symbol {
        "total-cost" => 'I',
        _ if symbol.starts_with("derived!") => 'D',
        _ => 'R',
    }
}

/// Python: def get_objects_by_type(typed_objects, types)
fn get_objects_by_type(
    objects: &[TypedObject],
    types: &[super::pddl::pddl_types::Type],
) -> HashMap<String, Vec<String>> {
    let mut result: HashMap<String, Vec<String>> = HashMap::new();
    for obj in objects {
        result
            .entry(obj.type_name.clone())
            .or_insert_with(Vec::new)
            .push(obj.name.clone());

        // Also add to supertypes
        let mut current_type = Some(obj.type_name.clone());
        while let Some(ref type_name) = current_type {
            if let Some(t) = types.iter().find(|t| &t.name == type_name) {
                if let Some(ref base) = t.basetype_name {
                    result
                        .entry(base.clone())
                        .or_insert_with(Vec::new)
                        .push(obj.name.clone());
                    current_type = Some(base.clone());
                } else {
                    current_type = None;
                }
            } else {
                current_type = None;
            }
        }
    }
    result
}

/// Python: def init_function_values(num_init)
fn init_function_values(
    num_init: &[FunctionAssignment],
) -> HashMap<PrimitiveNumericExpression, f64> {
    let mut result = HashMap::new();
    for assign in num_init {
        if let FunctionalExpression::NumericConstant(nc) = &assign.expression {
            result.insert(assign.fluent.clone(), nc.value.into_inner());
        }
    }
    result
}

/// Translates the task into a logic program, computes its model and
/// instantiates the actions and axioms the model proves reachable.
pub fn explore(task: &Task) -> ExploreResult {
    let prog = pddl_to_prolog::translate(task);
    let model = build_model::compute_model(&prog);

    let fluent_facts = get_fluent_facts(task);
    let objects_by_type = get_objects_by_type(&task.objects, &task.types);
    let init_func_vals = init_function_values(&task.num_init);
    let init_facts: HashSet<Atom> = task.init.iter().cloned().collect();

    let roles: Vec<Role> = model
        .symbols
        .predicates()
        .map(|(_, name)| Role::classify(name, &fluent_facts))
        .collect();

    let mut fluent_functions: HashSet<PrimitiveNumericExpression> = HashSet::new();
    let mut reachable_atoms: Vec<Atom> = vec![];
    let mut reachable_action_params: HashMap<String, Vec<Vec<String>>> = HashMap::new();
    let mut axiom_instances: Vec<(&str, Vec<String>)> = vec![];
    let mut relaxed_reachable = false;
    for atom in &model.atoms {
        let args = || -> Vec<String> {
            atom.args
                .iter()
                .map(|&arg| model.symbols.object_name(arg).to_owned())
                .collect()
        };
        let role = &roles[atom.predicate.index()];
        if role.fluent {
            reachable_atoms.push(Atom::new(
                model.symbols.predicate_name(atom.predicate).to_owned(),
                args(),
            ));
        }
        match role.bookkeeping {
            Bookkeeping::None => {}
            Bookkeeping::Function(symbol) => {
                fluent_functions.insert(PrimitiveNumericExpression::with_type(
                    symbol.to_owned(),
                    args(),
                    function_type(symbol),
                ));
            }
            Bookkeeping::ActionParameters(name) => reachable_action_params
                .entry(name.to_owned())
                .or_default()
                .push(args()),
            Bookkeeping::FunctionAxiom(name) => axiom_instances.push((name, args())),
            Bookkeeping::GoalReachable => relaxed_reachable = true,
        }
    }

    // Step 6: Instantiate actions
    let mut task_function_admin = task.function_administrator.clone();
    let mut grounded_ops: Vec<PropositionalAction> = vec![];
    let mut new_constant_numeric_axioms: Vec<InstantiatedNumericAxiom> = vec![];

    for action in &task.actions {
        if let Some(param_lists) = reachable_action_params.get(&action.name) {
            for params in param_lists {
                if params.len() == action.num_external_parameters {
                    let mut var_mapping: HashMap<String, String> = HashMap::new();
                    for (param, value) in action.parameters.iter().zip(params.iter()) {
                        var_mapping.insert(param.name.clone(), value.clone());
                    }
                    // For parameters beyond num_external, they're internal
                    // We need to handle them too
                    if let Some(prop_action) = action.instantiate(
                        &var_mapping,
                        &init_facts,
                        &fluent_facts,
                        &fluent_functions,
                        &init_func_vals,
                        &mut task_function_admin,
                        &mut new_constant_numeric_axioms,
                    ) {
                        grounded_ops.push(prop_action);
                    }
                }
            }
        }
    }

    // Step 7: Instantiate axioms
    let mut grounded_axioms: Vec<PropositionalAxiom> = vec![];
    for axiom in &task.axioms {
        // Try all possible parameter instantiations
        let param_combinations = get_parameter_combinations(&axiom.parameters, &objects_by_type);
        for params in &param_combinations {
            let mut var_mapping: HashMap<String, String> = HashMap::new();
            for (param, value) in axiom.parameters.iter().zip(params.iter()) {
                var_mapping.insert(param.name.clone(), value.clone());
            }
            if let Some(prop_axiom) = axiom.instantiate(
                &var_mapping,
                &init_facts,
                &fluent_facts,
                &fluent_functions,
                &init_func_vals,
                &mut task_function_admin,
                &mut new_constant_numeric_axioms,
            ) {
                grounded_axioms.push(prop_axiom);
            }
        }
    }

    let mut numeric_axioms: Vec<InstantiatedNumericAxiom> = vec![];
    let numeric_axioms_by_name: HashMap<String, super::pddl::axioms::NumericAxiom> =
        task_function_admin
            .get_all_axioms()
            .into_iter()
            .map(|axiom| (axiom.name.clone(), axiom))
            .collect();

    for (name, args) in &axiom_instances {
        if let Some(axiom) = numeric_axioms_by_name.get(*name) {
            let mut var_mapping: HashMap<String, String> = HashMap::new();
            for (parameter, value) in axiom.parameters.iter().zip(args) {
                var_mapping.insert(parameter.name.clone(), value.clone());
            }
            let instantiated = axiom.instantiate(
                &var_mapping,
                &fluent_functions,
                &init_func_vals,
                &mut task_function_admin,
                &mut new_constant_numeric_axioms,
            );
            if !numeric_axioms.contains(&instantiated) {
                numeric_axioms.push(instantiated);
            }
        }
    }

    let mut used_derived: HashSet<PrimitiveNumericExpression> = HashSet::new();
    for op in &grounded_ops {
        for cond in &op.precondition {
            collect_used_derived_pnes_from_condition(cond, &mut used_derived);
        }
        for (_, assign) in &op.assign_effects {
            if assign.fluent.symbol.starts_with("derived!") {
                used_derived.insert(assign.fluent.clone());
            }
            collect_used_derived_pnes_from_expr(&assign.expression, &mut used_derived);
        }
        if let Some(cost) = &op.cost {
            if cost.fluent.symbol.starts_with("derived!") {
                used_derived.insert(cost.fluent.clone());
            }
            collect_used_derived_pnes_from_expr(&cost.expression, &mut used_derived);
        }
    }
    for axiom in &grounded_axioms {
        for cond in &axiom.condition {
            collect_used_derived_pnes_from_condition(cond, &mut used_derived);
        }
    }

    for axiom in numeric_axioms_by_name.values() {
        let head = axiom.get_head();
        for used in used_derived
            .iter()
            .filter(|p| p.symbol == head.symbol && p.args.len() == axiom.parameters.len())
        {
            let mut var_mapping: HashMap<String, String> = HashMap::new();
            for (param, value) in axiom.parameters.iter().zip(used.args.iter()) {
                var_mapping.insert(param.name.clone(), value.clone());
            }
            let instantiated = axiom.instantiate(
                &var_mapping,
                &fluent_functions,
                &init_func_vals,
                &mut task_function_admin,
                &mut new_constant_numeric_axioms,
            );
            if !numeric_axioms.contains(&instantiated) {
                numeric_axioms.push(instantiated);
            }
        }
    }
    for axiom in new_constant_numeric_axioms {
        if !numeric_axioms.contains(&axiom) {
            numeric_axioms.push(axiom);
        }
    }

    // Step 9: Collect fluent numeric expressions.
    // Python returns the grounded fluent functions from the exploration model
    // here, not every declared or referenced numeric expression.
    let num_fluents: Vec<PrimitiveNumericExpression> = fluent_functions.iter().cloned().collect();

    // Determine initial constant predicates and numerics
    let mut init_constant_predicates: Vec<Atom> = vec![];
    let mut init_constant_numerics: Vec<FunctionAssignment> = vec![];
    for atom in &task.init {
        if !fluent_facts.contains(&atom.predicate) && !atom.predicate.starts_with("==") {
            init_constant_predicates.push(atom.clone());
        }
    }
    for assign in &task.num_init {
        if !fluent_functions.contains(&assign.fluent) {
            init_constant_numerics.push(assign.clone());
        }
    }

    ExploreResult {
        relaxed_reachable,
        atoms: reachable_atoms,
        num_fluents,
        grounded_ops,
        grounded_axioms,
        numeric_axioms,
        init_constant_predicates,
        init_constant_numerics,
        reachable_action_params,
    }
}

/// Python: explore_normalized is the same but takes NormalizableTask
pub fn explore_normalized(
    norm_task: &super::normalize::NormalizableTask,
) -> Result<ExploreResult, String> {
    Ok(explore(&norm_task.task))
}

/// Get all parameter combinations for a set of typed parameters.
fn get_parameter_combinations(
    parameters: &[TypedObject],
    objects_by_type: &HashMap<String, Vec<String>>,
) -> Vec<Vec<String>> {
    if parameters.is_empty() {
        return vec![vec![]];
    }

    let param = &parameters[0];
    let rest_combos = get_parameter_combinations(&parameters[1..], objects_by_type);

    let objects = objects_by_type
        .get(&param.type_name)
        .cloned()
        .unwrap_or_else(Vec::new);

    let mut result = vec![];
    for obj in &objects {
        for combo in &rest_combos {
            let mut new_combo = vec![obj.clone()];
            new_combo.extend(combo.clone());
            result.push(new_combo);
        }
    }
    result
}
