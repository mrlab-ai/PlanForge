/// Instantiates the PDDL task using the logic program model.
use std::collections::{BTreeMap, HashMap, HashSet};
use std::rc::Rc;

use super::build_model;
use super::grounding::{GroundingLimitError, GroundingLimits, GroundingMonitor};
use super::pddl::actions::PropositionalAction;
use super::pddl::axioms::{InstantiatedNumericAxiom, PropositionalAxiom};
use super::pddl::conditions::*;
use super::pddl::f_expression::*;
use super::pddl::pddl_types::TypedObject;
use super::pddl::tasks::{Task, VarMapping};
use super::pddl_to_prolog;
use super::symbols::ObjectId;
use super::tools::OrderedSet;

fn collect_used_derived_pnes_from_expr(
    expr: &FunctionalExpression,
    out: &mut OrderedSet<PrimitiveNumericExpression>,
) {
    if let FunctionalExpression::PrimitiveNumericExpression(pne) = expr {
        if pne.symbol.starts_with("derived!") {
            out.insert(pne.clone());
        }
        return;
    }
    for part in expr.parts() {
        collect_used_derived_pnes_from_expr(part, out);
    }
}

fn collect_used_derived_pnes_from_condition(
    cond: &Condition,
    out: &mut OrderedSet<PrimitiveNumericExpression>,
) {
    for operand in cond.comparison_operands() {
        collect_used_derived_pnes_from_expr(operand, out);
    }
    for part in cond.parts() {
        collect_used_derived_pnes_from_condition(part, out);
    }
}

/// Result of the exploration/instantiation process.
pub struct ExploreResult {
    pub atoms: Vec<Atom>,
    pub num_fluents: Vec<PrimitiveNumericExpression>,
    pub grounded_ops: Vec<PropositionalAction>,
    pub grounded_axioms: Vec<PropositionalAxiom>,
    pub numeric_axioms: Vec<InstantiatedNumericAxiom>,
    /// The parameter tuple of every reachable instance of each action, with
    /// the objects interned as the model left them. The invariant synthesis
    /// only asks whether two positions of a tuple hold the same object, so the
    /// tuples never have to be spelled back out.
    pub reachable_action_params: HashMap<String, Vec<Rc<[ObjectId]>>>,
}

/// A predicate is fluent if it appears in some action effect or is derived.
///
/// Which *atoms* of those predicates are fluent is a different question, decided
/// by the reachability model; see [`GroundingTables::fluent_facts`].
fn get_fluent_predicates(task: &Task) -> HashSet<String> {
    let mut fluent_preds: HashSet<String> = HashSet::new();
    for action in &task.actions {
        for eff in &action.effects {
            // Not `if let`: normalization leaves every effect a literal, and an
            // effect that is not one would drop out of this set silently. A
            // predicate missing here is treated as static, so its atoms never
            // become SAS variables and conditions on them are read as constants.
            // `invariant_finder::get_fluents` had the same hole.
            let pred = eff.peffect.literal_predicate().unwrap_or_else(|| {
                panic!(
                    "effect {} of action {} is not a literal after normalization",
                    eff.peffect, action.name
                )
            });
            fluent_preds.insert(pred.to_string());
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
    /// A reachable parameter tuple of the named propositional axiom.
    AxiomParameters(&'a str),
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
        } else if let Some(name) = predicate.strip_prefix("@axiom-") {
            Bookkeeping::AxiomParameters(name)
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

/// Lists every object of each type, counting an object as belonging to all of
/// its supertypes.
fn get_objects_by_type(
    objects: &[TypedObject],
    types: &[super::pddl::pddl_types::Type],
) -> HashMap<String, Vec<String>> {
    let supertype: HashMap<&str, &str> = types
        .iter()
        .filter_map(|t| Some((t.name.as_str(), t.basetype_name.as_deref()?)))
        .collect();

    let mut result: HashMap<String, Vec<String>> = types
        .iter()
        .map(|type_| (type_.name.clone(), Vec::new()))
        .chain(std::iter::once(("object".to_string(), Vec::new())))
        .collect();
    for object in objects {
        let mut type_name = object.type_name.as_str();
        loop {
            result
                .entry(type_name.to_owned())
                .or_default()
                .push(object.name.clone());
            match supertype.get(type_name) {
                Some(base) => type_name = base,
                None => break,
            }
        }
    }
    result
}

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
pub fn explore(task: &Task, limits: GroundingLimits) -> Result<ExploreResult, GroundingLimitError> {
    let mut monitor = GroundingMonitor::new(limits);
    let prog = pddl_to_prolog::translate(task);
    let model = build_model::compute_model(&prog, &mut monitor)?;
    monitor.enter_phase("collecting reachable facts");

    let fluent_predicates = get_fluent_predicates(task);
    let objects_by_type = get_objects_by_type(&task.objects, &task.types);
    let init_func_vals = init_function_values(&task.num_init);
    let init_facts: HashSet<Atom> = task.init.iter().cloned().collect();

    let roles: Vec<Role> = model
        .symbols
        .predicates()
        .map(|(_, name)| Role::classify(name, &fluent_predicates))
        .collect();

    let mut fluent_functions: HashSet<PrimitiveNumericExpression> = HashSet::new();
    let mut reachable_atoms: Vec<Atom> = vec![];
    let mut reachable_action_params: HashMap<String, Vec<Rc<[ObjectId]>>> = HashMap::new();
    let mut reachable_axiom_params: HashMap<(String, usize), Vec<Rc<[ObjectId]>>> = HashMap::new();
    let mut axiom_instances: Vec<(&str, Vec<String>)> = vec![];
    for atom in &model.atoms {
        let args = || -> Vec<String> {
            atom.args
                .iter()
                .map(|&arg| model.symbols.object_name(arg).to_owned())
                .collect()
        };
        let role = &roles[atom.predicate.index()];
        if role.fluent {
            let predicate = model.symbols.predicate_name(atom.predicate);
            let arguments = args();
            let string_bytes = predicate.len() + arguments.iter().map(String::len).sum::<usize>();
            monitor.note_materialized_bytes(
                (std::mem::size_of::<Atom>()
                    + arguments.len() * std::mem::size_of::<String>()
                    + string_bytes) as u64,
            )?;
            reachable_atoms.push(Atom::new(predicate.to_owned(), arguments));
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
            Bookkeeping::ActionParameters(name) => {
                monitor.note_materialized_bytes(std::mem::size_of::<Rc<[ObjectId]>>() as u64)?;
                reachable_action_params
                    .entry(name.to_owned())
                    .or_default()
                    .push(Rc::clone(&atom.args));
            }
            Bookkeeping::AxiomParameters(name) => {
                monitor.note_materialized_bytes(std::mem::size_of::<Rc<[ObjectId]>>() as u64)?;
                reachable_axiom_params
                    .entry((name.to_owned(), atom.args.len()))
                    .or_default()
                    .push(Rc::clone(&atom.args));
            }
            Bookkeeping::FunctionAxiom(name) => axiom_instances.push((name, args())),
            // The goal atom being derived means the delete-relaxation reaches
            // the goal. Nothing here needs that: the translation of the
            // grounded task already answers with a trivial task when the goal
            // is out of reach, and it does so on the real task rather than the
            // relaxation.
            Bookkeeping::GoalReachable => {}
        }
    }

    // HashSet owns deep clones of every reachable atom.
    let fluent_fact_bytes = reachable_atoms
        .iter()
        .map(|atom| {
            std::mem::size_of::<Atom>()
                + atom.args.len() * std::mem::size_of::<String>()
                + atom.predicate.len()
                + atom.args.iter().map(String::len).sum::<usize>()
                + 16
        })
        .sum::<usize>();
    monitor.note_materialized_bytes(fluent_fact_bytes as u64)?;
    let fluent_facts: HashSet<Atom> = reachable_atoms.iter().cloned().collect();
    let tables = &super::pddl::tasks::GroundingTables {
        init_facts: &init_facts,
        fluent_facts: &fluent_facts,
        fluent_functions: &fluent_functions,
        init_function_vals: &init_func_vals,
        objects_by_type: &objects_by_type,
    };

    // Step 6: Instantiate actions
    let mut task_function_admin = task.function_administrator.clone();
    let mut grounded_ops: Vec<PropositionalAction> = vec![];
    let mut new_constant_numeric_axioms: OrderedSet<InstantiatedNumericAxiom> =
        OrderedSet::default();

    // The mapping is rebuilt for every instance and borrows the names it maps,
    // so one buffer serves all of them.
    let mut var_mapping = VarMapping::default();
    for action in &task.actions {
        let Some(param_lists) = reachable_action_params.get(&action.name) else {
            continue;
        };
        monitor.enter_action(&action.name);
        for params in param_lists {
            // The exploration rule for an action has `action.parameters` as its
            // head arguments, so a reachable tuple has exactly that length. It is
            // longer than `num_external_parameters` whenever an existential
            // precondition contributed variables: eliminating the quantifier moves
            // them into the parameter list and deliberately leaves the external
            // count alone, because the external ones are what name the action.
            //
            // This used to skip every tuple longer than the external count, which
            // silently deleted every action with an `exists` precondition. Any
            // other length is an inconsistency between rule generation and
            // grounding, so it fails here rather than dropping an action.
            assert_eq!(
                params.len(),
                action.parameters.len(),
                "action {} was explored with {} arguments but declares {} parameters",
                action.name,
                params.len(),
                action.parameters.len()
            );
            var_mapping.clear();
            for (parameter, &object) in action.parameters.iter().zip(params.iter()) {
                var_mapping.bind(&parameter.name, model.symbols.object_name(object));
            }
            if let Some(prop_action) = action.instantiate(
                &var_mapping,
                tables,
                &mut task_function_admin,
                &mut new_constant_numeric_axioms,
            ) {
                let estimated_bytes = std::mem::size_of::<PropositionalAction>()
                    + prop_action.name.len()
                    + prop_action.precondition.len() * 128
                    + (prop_action.add_effects.len() + prop_action.del_effects.len()) * 192
                    + prop_action.assign_effects.len() * 256;
                monitor.note_action(estimated_bytes as u64)?;
                grounded_ops.push(prop_action);
            }
        }
    }

    // Step 7: Instantiate axioms
    monitor.enter_phase("instantiating axioms");
    let mut grounded_axioms: Vec<PropositionalAxiom> = vec![];
    let mut var_mapping = VarMapping::default();
    for axiom in &task.axioms {
        let Some(param_lists) =
            reachable_axiom_params.get(&(axiom.name.clone(), axiom.parameters.len()))
        else {
            continue;
        };
        for params in param_lists {
            assert_eq!(
                params.len(),
                axiom.parameters.len(),
                "axiom {} was explored with {} arguments but declares {} parameters",
                axiom.name,
                params.len(),
                axiom.parameters.len()
            );
            var_mapping.clear();
            for (parameter, &object) in axiom.parameters.iter().zip(params.iter()) {
                var_mapping.bind(&parameter.name, model.symbols.object_name(object));
            }
            if let Some(prop_axiom) = axiom.instantiate(
                &var_mapping,
                tables,
                &mut task_function_admin,
                &mut new_constant_numeric_axioms,
            ) {
                grounded_axioms.push(prop_axiom);
            }
        }
    }

    let mut numeric_axioms: OrderedSet<InstantiatedNumericAxiom> = OrderedSet::default();
    // Ordered by name: which of two equivalent axioms is instantiated first
    // decides which one names a SAS numeric variable.
    let numeric_axioms_by_name: BTreeMap<String, super::pddl::axioms::NumericAxiom> =
        task_function_admin
            .get_all_axioms()
            .into_iter()
            .map(|axiom| (axiom.name.clone(), axiom))
            .collect();

    let mut var_mapping = VarMapping::default();
    for (name, args) in &axiom_instances {
        if let Some(axiom) = numeric_axioms_by_name.get(*name) {
            var_mapping.clear();
            for (parameter, value) in axiom.parameters.iter().zip(args) {
                var_mapping.bind(&parameter.name, value);
            }
            let instantiated = axiom.instantiate(
                &var_mapping,
                tables.fluent_functions,
                tables.init_function_vals,
                &mut task_function_admin,
                &mut new_constant_numeric_axioms,
            );
            numeric_axioms.insert(instantiated);
        }
    }

    let mut used_derived: OrderedSet<PrimitiveNumericExpression> = OrderedSet::default();
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

    let used_derived = used_derived.into_vec();
    let mut used_derived_by_symbol: HashMap<&str, Vec<&PrimitiveNumericExpression>> =
        HashMap::new();
    for expression in &used_derived {
        used_derived_by_symbol
            .entry(&expression.symbol)
            .or_default()
            .push(expression);
    }
    for axiom in numeric_axioms_by_name.values() {
        let head = axiom.get_head();
        let matching_arity = used_derived_by_symbol
            .get(head.symbol.as_str())
            .map_or(&[][..], Vec::as_slice)
            .iter()
            .copied()
            .filter(|expression| expression.args.len() == axiom.parameters.len());
        for used in matching_arity {
            var_mapping.clear();
            for (parameter, value) in axiom.parameters.iter().zip(used.args.iter()) {
                var_mapping.bind(&parameter.name, value);
            }
            let instantiated = axiom.instantiate(
                &var_mapping,
                tables.fluent_functions,
                tables.init_function_vals,
                &mut task_function_admin,
                &mut new_constant_numeric_axioms,
            );
            numeric_axioms.insert(instantiated);
        }
    }
    for axiom in new_constant_numeric_axioms.into_vec() {
        numeric_axioms.insert(axiom);
    }

    // Step 9: Collect fluent numeric expressions.
    // Python returns the grounded fluent functions from the exploration model
    // here, not every declared or referenced numeric expression.
    let num_fluents: Vec<PrimitiveNumericExpression> = fluent_functions.iter().cloned().collect();

    monitor.complete()?;
    Ok(ExploreResult {
        atoms: reachable_atoms,
        num_fluents,
        grounded_ops,
        grounded_axioms,
        numeric_axioms: numeric_axioms.into_vec(),
        reachable_action_params,
    })
}

/// Visits every assignment of type-correct objects to `parameters`, last
/// parameter varying fastest. The tuples are visited rather than collected:
/// there is one per object combination, and an axiom with three parameters
/// over a few hundred objects has more of them than fit in memory comfortably.
pub(crate) fn for_each_parameter_tuple<'a>(
    parameters: &[TypedObject],
    objects_by_type: &'a HashMap<String, Vec<String>>,
    visit: &mut impl FnMut(&[&'a str]),
) {
    let domains: Vec<&[String]> = parameters
        .iter()
        .map(|parameter| {
            objects_by_type
                .get(&parameter.type_name)
                .unwrap_or_else(|| {
                    panic!(
                        "parameter {} uses type {} that was not validated at parse time",
                        parameter.name, parameter.type_name
                    )
                })
                .as_slice()
        })
        .collect();
    if domains.iter().any(|objects| objects.is_empty()) {
        return;
    }

    let mut cursor = vec![0usize; domains.len()];
    let mut tuple: Vec<&str> = domains.iter().map(|objects| objects[0].as_str()).collect();
    loop {
        visit(&tuple);
        let mut level = domains.len();
        loop {
            if level == 0 {
                return;
            }
            level -= 1;
            cursor[level] += 1;
            if cursor[level] < domains[level].len() {
                tuple[level] = domains[level][cursor[level]].as_str();
                break;
            }
            cursor[level] = 0;
            tuple[level] = domains[level][0].as_str();
        }
    }
}
