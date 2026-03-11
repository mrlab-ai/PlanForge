/// Port of pddl/actions.py
use std::collections::{HashMap, HashSet};
use std::fmt;

use super::conditions::{Atom, Condition, Conjunction, NegatedAtom};
use super::effects::{Effect, EffectKind, EffectType};
use super::f_expression::{
    FunctionAssignment, FunctionalExpression, NumericConstant, PrimitiveNumericExpression,
};
use super::pddl_types::TypedObject;

/// Python: class Action(object)
#[derive(Debug, Clone)]
pub struct Action {
    pub name: String,
    pub parameters: Vec<TypedObject>,
    pub num_external_parameters: usize,
    pub precondition: Condition,
    pub effects: Vec<Effect>,
    pub cost: Option<FunctionAssignment>,
    /// Numeric (assign) effects: parameters, condition, assignment
    pub assign_effects: Vec<(Vec<TypedObject>, Condition, FunctionAssignment)>,
}

impl Action {
    pub fn new(
        name: String,
        parameters: Vec<TypedObject>,
        num_external_parameters: usize,
        precondition: Condition,
        effects: Vec<Effect>,
        cost: Option<FunctionAssignment>,
    ) -> Self {
        Action {
            name,
            parameters,
            num_external_parameters,
            precondition,
            effects,
            cost,
            assign_effects: vec![],
        }
    }

    /// Python: def parse(alist)
    /// Parsing is handled in pddl_parser/parsing_functions.rs

    /// Python: def uniquify_variables(self)
    pub fn uniquify_variables(&mut self) {
        let mut type_map: HashMap<String, usize> = HashMap::new();
        let mut renamings: HashMap<String, String> = HashMap::new();
        for p in &mut self.parameters {
            p.uniquify_name(&mut type_map, &mut renamings);
        }
        self.precondition = self
            .precondition
            .uniquify_variables(&mut type_map, &mut renamings);
        // Effects already have their own parameters that need renaming
        let mut new_effects = vec![];
        for eff in &self.effects {
            let mut new_params = eff.parameters.clone();
            for p in &mut new_params {
                p.uniquify_name(&mut type_map, &mut renamings);
            }
            let new_cond = eff
                .condition
                .uniquify_variables(&mut type_map, &mut renamings);
            let new_peff = eff
                .peffect
                .uniquify_variables(&mut type_map, &mut renamings);
            new_effects.push(Effect::new(new_params, new_cond, new_peff));
        }
        self.effects = new_effects;

        let mut new_assign_effects = vec![];
        for (params, condition, assignment) in &self.assign_effects {
            let mut new_params = params.clone();
            for p in &mut new_params {
                p.uniquify_name(&mut type_map, &mut renamings);
            }
            let new_cond = condition.uniquify_variables(&mut type_map, &mut renamings);
            let new_assign = assignment.rename_variables(&renamings);
            new_assign_effects.push((new_params, new_cond, new_assign));
        }
        self.assign_effects = new_assign_effects;
    }

    /// Python: def dump(self)
    pub fn dump(&self) {
        println!("Action {} ({} params)", self.name, self.parameters.len());
        println!("  precondition: {}", self.precondition);
        for eff in &self.effects {
            eff.dump();
        }
        if let Some(ref cost) = self.cost {
            println!("  cost: {}", cost);
        }
    }

    /// Python: def instantiate(self, var_mapping, init_facts, fluent_facts, init_function_vals, fluent_functions, task, new_axiom, new_modules)
    /// Returns a PropositionalAction or None if the precondition is statically false.
    pub fn instantiate(
        &self,
        var_mapping: &HashMap<String, String>,
        init_facts: &HashSet<Atom>,
        fluent_facts: &HashSet<String>,
        fluent_functions: &HashSet<PrimitiveNumericExpression>,
        init_function_vals: &HashMap<PrimitiveNumericExpression, f64>,
        task_function_admin: &mut super::tasks::DerivedFunctionAdministrator,
        new_constant_axioms: &mut Vec<super::axioms::InstantiatedNumericAxiom>,
    ) -> Option<PropositionalAction> {
        // Build the action name
        let arg_list: Vec<String> = self.parameters[..self.num_external_parameters]
            .iter()
            .map(|p| {
                var_mapping
                    .get(&p.name)
                    .cloned()
                    .unwrap_or_else(|| p.name.clone())
            })
            .collect();
        let name = format!("({} {})", self.name, arg_list.join(" "));

        // Instantiate precondition
        let mut precondition = vec![];
        match self.precondition.instantiate_action(
            var_mapping,
            init_facts,
            fluent_facts,
            fluent_functions,
            init_function_vals,
            task_function_admin,
            new_constant_axioms,
        ) {
            Some(conds) => precondition = conds,
            None => return None, // Precondition statically false
        }

        // Instantiate effects
        let mut add_effects = vec![];
        let mut del_effects = vec![];
        let mut assign_effects = vec![];

        for eff in &self.effects {
            // Check effect condition
            let eff_condition = match eff.condition.instantiate_action(
                var_mapping,
                init_facts,
                fluent_facts,
                fluent_functions,
                init_function_vals,
                task_function_admin,
                new_constant_axioms,
            ) {
                Some(conds) => conds,
                None => continue, // Effect condition statically false
            };

            match &eff.peffect {
                Condition::Atom(atom) => {
                    let new_args: Vec<String> = atom
                        .args
                        .iter()
                        .map(|a| var_mapping.get(a).cloned().unwrap_or_else(|| a.clone()))
                        .collect();
                    let new_atom = Atom::new(atom.predicate.clone(), new_args);
                    add_effects.push((eff_condition, new_atom));
                }
                Condition::NegatedAtom(natom) => {
                    let new_args: Vec<String> = natom
                        .args
                        .iter()
                        .map(|a| var_mapping.get(a).cloned().unwrap_or_else(|| a.clone()))
                        .collect();
                    let new_atom = Atom::new(natom.predicate.clone(), new_args);
                    del_effects.push((eff_condition, new_atom));
                }
                _ => panic!("Unexpected effect type in action instantiation"),
            }
        }

        for (params, condition, assignment) in &self.assign_effects {
            let mut eff_var_mapping = var_mapping.clone();
            for parameter in params {
                eff_var_mapping
                    .entry(parameter.name.clone())
                    .or_insert_with(|| parameter.name.clone());
            }
            let eff_condition = match condition.instantiate_action(
                &eff_var_mapping,
                init_facts,
                fluent_facts,
                fluent_functions,
                init_function_vals,
                task_function_admin,
                new_constant_axioms,
            ) {
                Some(conds) => conds,
                None => continue,
            };
            let instantiated_assignment = assignment.instantiate(
                &eff_var_mapping,
                fluent_functions,
                init_function_vals,
                task_function_admin,
                new_constant_axioms,
            );
            assign_effects.push((eff_condition, instantiated_assignment));
        }

        // Instantiate cost
        let cost = if let Some(ref c) = self.cost {
            Some(c.instantiate(
                var_mapping,
                fluent_functions,
                init_function_vals,
                task_function_admin,
                new_constant_axioms,
            ))
        } else {
            // Default cost: increase(total-cost, 1)
            let constant_expr = FunctionalExpression::NumericConstant(NumericConstant::new(1.0));
            let derived =
                task_function_admin.get_derived_function(&constant_expr, fluent_functions);
            if let Some(axiom) = task_function_admin
                .get_all_axioms()
                .into_iter()
                .find(|axiom| axiom.name == derived.symbol)
            {
                let instantiated_axiom = axiom.instantiate(
                    &HashMap::new(),
                    fluent_functions,
                    init_function_vals,
                    task_function_admin,
                    new_constant_axioms,
                );
                if !new_constant_axioms.contains(&instantiated_axiom) {
                    new_constant_axioms.push(instantiated_axiom);
                }
            }
            Some(FunctionAssignment::new(
                "+".to_string(),
                PrimitiveNumericExpression::with_type("total-cost".to_string(), vec![], 'I'),
                FunctionalExpression::PrimitiveNumericExpression(derived),
            ))
        };

        Some(PropositionalAction {
            name,
            precondition,
            add_effects,
            del_effects,
            assign_effects,
            cost,
        })
    }
}

/// Python: class PropositionalAction(object)
/// A ground action with propositional preconditions and effects.
#[derive(Debug, Clone)]
pub struct PropositionalAction {
    pub name: String,
    pub precondition: Vec<Condition>,
    pub add_effects: Vec<(Vec<Condition>, Atom)>,
    pub del_effects: Vec<(Vec<Condition>, Atom)>,
    pub assign_effects: Vec<(Vec<Condition>, FunctionAssignment)>,
    pub cost: Option<FunctionAssignment>,
}

impl PropositionalAction {
    pub fn new(
        name: String,
        precondition: Vec<Condition>,
        add_effects: Vec<(Vec<Condition>, Atom)>,
        del_effects: Vec<(Vec<Condition>, Atom)>,
        assign_effects: Vec<(Vec<Condition>, FunctionAssignment)>,
        cost: Option<FunctionAssignment>,
    ) -> Self {
        PropositionalAction {
            name,
            precondition,
            add_effects,
            del_effects,
            assign_effects,
            cost,
        }
    }

    pub fn dump(&self) {
        println!("PropositionalAction {}", self.name);
        println!("  Preconditions:");
        for p in &self.precondition {
            println!("    {}", p);
        }
        println!("  Add effects:");
        for (cond, atom) in &self.add_effects {
            println!("    {} <- {:?}", atom, cond);
        }
        println!("  Del effects:");
        for (cond, atom) in &self.del_effects {
            println!("    {} <- {:?}", atom, cond);
        }
        println!("  Assign effects:");
        for (cond, assign) in &self.assign_effects {
            println!("    {} <- {:?}", assign, cond);
        }
    }
}

// Add instantiate_action method to Condition
impl Condition {
    /// Instantiate a condition for action instantiation.
    /// Returns None if the condition is statically false.
    /// Returns Some(vec![]) if statically true.
    /// Returns Some(conditions) for the fluent conditions.
    pub fn instantiate_action(
        &self,
        var_mapping: &HashMap<String, String>,
        init_facts: &HashSet<Atom>,
        fluent_facts: &HashSet<String>,
        fluent_functions: &HashSet<PrimitiveNumericExpression>,
        init_function_vals: &HashMap<PrimitiveNumericExpression, f64>,
        task_function_admin: &mut super::tasks::DerivedFunctionAdministrator,
        new_constant_axioms: &mut Vec<super::axioms::InstantiatedNumericAxiom>,
    ) -> Option<Vec<Condition>> {
        let mut result = vec![];
        match self {
            Condition::Truth => Some(vec![]),
            Condition::Falsity => None,
            Condition::Conjunction(conj) => {
                for part in &conj.parts {
                    match part.instantiate_action(
                        var_mapping,
                        init_facts,
                        fluent_facts,
                        fluent_functions,
                        init_function_vals,
                        task_function_admin,
                        new_constant_axioms,
                    ) {
                        Some(conds) => result.extend(conds),
                        None => return None,
                    }
                }
                Some(result)
            }
            Condition::Atom(atom) => {
                let new_args: Vec<String> = atom
                    .args
                    .iter()
                    .map(|a| var_mapping.get(a).cloned().unwrap_or_else(|| a.clone()))
                    .collect();
                let new_atom = Atom::new(atom.predicate.clone(), new_args);
                if fluent_facts.contains(&atom.predicate) {
                    Some(vec![Condition::Atom(new_atom)])
                } else if init_facts.contains(&new_atom) {
                    Some(vec![]) // static true
                } else {
                    None // static false
                }
            }
            Condition::NegatedAtom(natom) => {
                let new_args: Vec<String> = natom
                    .args
                    .iter()
                    .map(|a| var_mapping.get(a).cloned().unwrap_or_else(|| a.clone()))
                    .collect();
                let pos_atom = Atom::new(natom.predicate.clone(), new_args.clone());
                if fluent_facts.contains(&natom.predicate) {
                    Some(vec![Condition::NegatedAtom(NegatedAtom::new(
                        natom.predicate.clone(),
                        new_args,
                    ))])
                } else if init_facts.contains(&pos_atom) {
                    None // static true, but we need it negated -> false
                } else {
                    Some(vec![]) // static false, negation is true
                }
            }
            Condition::FunctionComparison(fc) => {
                // Instantiate the function comparison
                let new_parts = fc
                    .parts
                    .iter()
                    .map(|p| {
                        super::f_expression::instantiate_expression(
                            p,
                            var_mapping,
                            fluent_functions,
                            init_function_vals,
                            task_function_admin,
                            new_constant_axioms,
                        )
                    })
                    .collect();
                let new_fc =
                    super::conditions::FunctionComparison::new(fc.comparator.clone(), new_parts);
                Some(vec![Condition::FunctionComparison(
                    super::conditions::FunctionComparison::new(new_fc.comparator, new_fc.parts),
                )])
            }
            Condition::NegatedFunctionComparison(nfc) => {
                let new_parts = nfc
                    .parts
                    .iter()
                    .map(|p| {
                        super::f_expression::instantiate_expression(
                            p,
                            var_mapping,
                            fluent_functions,
                            init_function_vals,
                            task_function_admin,
                            new_constant_axioms,
                        )
                    })
                    .collect();
                let new_nfc = super::conditions::NegatedFunctionComparison::new(
                    nfc.comparator.clone(),
                    new_parts,
                );
                Some(vec![Condition::NegatedFunctionComparison(
                    super::conditions::NegatedFunctionComparison::new(
                        new_nfc.comparator,
                        new_nfc.parts,
                    ),
                )])
            }
            _ => {
                // For other condition types, just return them as-is
                Some(vec![self.clone()])
            }
        }
    }
}
