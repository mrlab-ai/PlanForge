use std::collections::HashMap;

use super::conditions::{Atom, Condition};
use super::effects::Effect;
use super::f_expression::{
    FunctionAssignment, FunctionalExpression, NumericConstant, PrimitiveNumericExpression,
};
use super::pddl_types::TypedObject;

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

    /// Returns a PropositionalAction or None if the precondition is statically false.
    pub fn instantiate(
        &self,
        var_mapping: &super::tasks::VarMapping,
        tables: &super::tasks::GroundingTables,
        task_function_admin: &mut super::tasks::DerivedFunctionAdministrator,
        new_constant_axioms: &mut Vec<super::axioms::InstantiatedNumericAxiom>,
    ) -> Option<PropositionalAction> {
        let super::tasks::GroundingTables {
            fluent_functions,
            init_function_vals,
            ..
        } = *tables;
        // Build the action name
        let arg_list: Vec<&str> = self.parameters[..self.num_external_parameters]
            .iter()
            .map(|parameter| crate::pddl::Substitution::resolve(var_mapping, &parameter.name))
            .collect();
        let name = format!("({} {})", self.name, arg_list.join(" "));

        // Instantiate precondition
        let mut precondition = vec![];
        {
            let conds = self.precondition.instantiate_action(
                var_mapping,
                tables,
                task_function_admin,
                new_constant_axioms,
            )?;
            precondition = conds
        }

        // Instantiate effects
        let mut add_effects = vec![];
        let mut del_effects = vec![];
        let mut assign_effects = vec![];

        for eff in &self.effects {
            // Check effect condition
            let eff_condition = match eff.condition.instantiate_action(
                var_mapping,
                tables,
                task_function_admin,
                new_constant_axioms,
            ) {
                Some(conds) => conds,
                None => continue, // Effect condition statically false
            };

            match &eff.peffect {
                Condition::Atom(atom) => {
                    add_effects.push((eff_condition, atom.substituted(var_mapping)));
                }
                // A delete effect is recorded by the atom it removes, so the
                // substitution goes straight into that atom: negating the
                // literal first would copy the arguments a second time, once per
                // delete effect of every reachable instance of every action.
                Condition::NegatedAtom(natom) => del_effects.push((
                    eff_condition,
                    Atom::new(
                        natom.predicate.clone(),
                        crate::pddl::substitute(&natom.args, var_mapping),
                    ),
                )),
                _ => panic!("Unexpected effect type in action instantiation"),
            }
        }

        for (params, condition, assignment) in &self.assign_effects {
            let mut eff_var_mapping = var_mapping.clone();
            for parameter in params {
                eff_var_mapping.bind_to_itself(&parameter.name);
            }
            let eff_condition = match condition.instantiate_action(
                &eff_var_mapping,
                tables,
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
            let derived = task_function_admin.get_derived_function(&constant_expr);
            if let Some(axiom) = task_function_admin.axiom_named(&derived.symbol).cloned() {
                let instantiated_axiom = axiom.instantiate(
                    &super::tasks::VarMapping::default(),
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

// Add instantiate_action method to Condition
impl Condition {
    /// Instantiate a condition for action instantiation.
    /// Returns None if the condition is statically false.
    /// Returns Some(vec![]) if statically true.
    /// Returns Some(conditions) for the fluent conditions.
    pub fn instantiate_action(
        &self,
        var_mapping: &super::tasks::VarMapping,
        tables: &super::tasks::GroundingTables,
        task_function_admin: &mut super::tasks::DerivedFunctionAdministrator,
        new_constant_axioms: &mut Vec<super::axioms::InstantiatedNumericAxiom>,
    ) -> Option<Vec<Condition>> {
        let super::tasks::GroundingTables {
            init_facts,
            fluent_facts,
            fluent_functions,
            init_function_vals,
        } = *tables;
        let mut result = vec![];
        match self {
            Condition::Truth => Some(vec![]),
            Condition::Falsity => None,
            Condition::Conjunction(conj) => {
                for part in &conj.parts {
                    {
                        let conds = part.instantiate_action(
                            var_mapping,
                            tables,
                            task_function_admin,
                            new_constant_axioms,
                        )?;
                        result.extend(conds)
                    }
                }
                Some(result)
            }
            // A reachable fluent atom stays a condition; anything else is
            // decided here, at the only point where the reachable set is known.
            // Note that both arms test the *atom*, not its predicate: an
            // unreachable instance of a fluent predicate is statically false.
            Condition::Atom(atom) => {
                let new_atom = atom.substituted(var_mapping);
                if fluent_facts.contains(&new_atom) {
                    Some(vec![Condition::Atom(new_atom)])
                } else if init_facts.contains(&new_atom) {
                    Some(vec![]) // statically true
                } else {
                    None // statically false
                }
            }
            Condition::NegatedAtom(natom) => {
                // Both tests are on the atom this literal denies, and only the
                // fluent case needs the literal itself, so the negated copy of
                // the arguments is made there and not before.
                let pos_atom = Atom::new(
                    natom.predicate.clone(),
                    crate::pddl::substitute(&natom.args, var_mapping),
                );
                if fluent_facts.contains(&pos_atom) {
                    Some(vec![Condition::NegatedAtom(pos_atom.negate())])
                } else if init_facts.contains(&pos_atom) {
                    None // statically true, and we need it negated
                } else {
                    Some(vec![]) // statically false, so its negation holds
                }
            }
            Condition::FunctionComparison(_) | Condition::NegatedFunctionComparison(_) => {
                Some(vec![self.map_comparison_operands(|operand| {
                    super::f_expression::instantiate_expression(
                        operand,
                        var_mapping,
                        fluent_functions,
                        init_function_vals,
                        task_function_admin,
                        new_constant_axioms,
                    )
                })])
            }
            _ => {
                // For other condition types, just return them as-is
                Some(vec![self.clone()])
            }
        }
    }
}
