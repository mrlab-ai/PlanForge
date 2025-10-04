//! PDDL effects representation  
//! Port of python/translate/pddl/effects.py

use std::collections::HashMap;
use super::{Condition, Literal};
use super::pddl_types::TypedObject;
use super::f_expression::FunctionExpression;

/// Cartesian product utility function for effect instantiation
pub fn cartesian_product<T: Clone>(sequences: &[Vec<T>]) -> Vec<Vec<T>> {
    if sequences.is_empty() {
        return vec![vec![]];
    }
    
    let mut result = vec![];
    let rest_product = cartesian_product(&sequences[1..]);
    
    for item in &sequences[0] {
        for rest in &rest_product {
            let mut new_tuple = vec![item.clone()];
            new_tuple.extend(rest.clone());
            result.push(new_tuple);
        }
    }
    
    result
}

#[derive(Debug, Clone, PartialEq)]
pub struct Effect {
    pub parameters: Vec<TypedObject>,
    pub condition: Condition,
    pub peffect: PrimitiveEffect,
}

impl Effect {
    pub fn new(parameters: Vec<TypedObject>, condition: Condition, peffect: PrimitiveEffect) -> Self {
        Self { parameters, condition, peffect }
    }

    pub fn dump(&self) {
        let mut indent = "  ";
        if !self.parameters.is_empty() {
            let params: Vec<String> = self.parameters.iter().map(|p| p.to_string()).collect();
            println!("{}forall {}", indent, params.join(", "));
            indent = "    ";
        }
        
        // Check if condition is Truth (simplified check)
        match &self.condition {
            Condition::Truth => {},
            _ => {
                println!("{}if", indent);
                // TODO: Implement condition dump
                println!("{}  (condition)", indent);
                println!("{}then", indent);
                indent = "    ";
            }
        }
        
        println!("{}{:?}", indent, self.peffect);
    }

    pub fn copy_effect(&self) -> Self {
        Self {
            parameters: self.parameters.clone(),
            condition: self.condition.clone(),
            peffect: self.peffect.clone(),
        }
    }

    pub fn uniquify_variables(&mut self, type_map: &mut HashMap<String, String>) {
        let mut renamings = HashMap::new();
        self.parameters = self.parameters
            .iter()
            .map(|par| par.uniquify_name(type_map, &mut renamings))
            .collect();
        
        // TODO: Implement condition variable uniquification
        // self.condition = self.condition.uniquify_variables(type_map, renamings);
        // self.peffect = self.peffect.rename_variables(renamings);
    }

    pub fn instantiate(
        &self,
        var_mapping: &HashMap<String, String>,
        init_facts: &[Literal],
        fluent_facts: &[Literal],
        init_function_vals: &HashMap<String, f64>,
        fluent_functions: &[String],
        // TODO: Add task, new_axiom, new_modules parameters
        objects_by_type: &HashMap<String, Vec<String>>,
        result: &mut Vec<(Vec<Literal>, PrimitiveEffect)>,
    ) {
        if !self.parameters.is_empty() {
            let mut var_mapping = var_mapping.clone();
            let object_lists: Vec<Vec<String>> = self.parameters
                .iter()
                .map(|par| objects_by_type.get(par.get_type_name()).cloned().unwrap_or_default())
                .collect();
                
            for object_tuple in cartesian_product(&object_lists) {
                for (par, obj) in self.parameters.iter().zip(&object_tuple) {
                    var_mapping.insert(par.name.clone(), obj.clone());
                }
                self._instantiate(&var_mapping, init_facts, fluent_facts, 
                                init_function_vals, fluent_functions, objects_by_type, result);
            }
        } else {
            self._instantiate(var_mapping, init_facts, fluent_facts, 
                            init_function_vals, fluent_functions, objects_by_type, result);
        }
    }

    fn _instantiate(
        &self,
        _var_mapping: &HashMap<String, String>,
        _init_facts: &[Literal],
        _fluent_facts: &[Literal],
        _init_function_vals: &HashMap<String, f64>,
        _fluent_functions: &[String],
        _objects_by_type: &HashMap<String, Vec<String>>,
        result: &mut Vec<(Vec<Literal>, PrimitiveEffect)>,
    ) {
        let condition = vec![];
        
        // TODO: Implement condition instantiation
        // try {
        //     self.condition.instantiate(var_mapping, init_facts, fluent_facts, 
        //                               init_function_vals, fluent_functions, task, 
        //                               new_axiom, new_modules, condition);
        // } catch Impossible => return;
        
        let mut effects = vec![];
        // TODO: Implement peffect instantiation
        // self.peffect.instantiate(var_mapping, init_facts, fluent_facts,
        //                         init_function_vals, fluent_functions, task,
        //                         new_axiom, new_modules, effects);
        
        // For now, add a placeholder
        effects.push(self.peffect.clone());
        
        if !effects.is_empty() {
            result.push((condition, effects[0].clone()));
        }
    }

    pub fn relaxed(&self) -> Option<Self> {
        match &self.peffect {
            PrimitiveEffect::Literal(lit) if lit.negated => None,
            _ => Some(Self {
                parameters: self.parameters.clone(),
                condition: self.condition.clone(), // TODO: Implement condition.relaxed()
                peffect: self.peffect.clone(),
            })
        }
    }

    pub fn simplified(self) -> Self {
        Self {
            parameters: self.parameters,
            condition: self.condition.simplified(),
            peffect: self.peffect,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum PrimitiveEffect {
    // New structured variants
    Literal(Literal),
    FunctionAssignment(FunctionAssignment),
    
    // Backward compatibility variants for porting  
    Add(String, Vec<String>),
    Del(String, Vec<String>),
    Increase(String, Vec<String>, f64),
    Decrease(String, Vec<String>, f64),
}

#[derive(Debug, Clone, PartialEq)]
pub struct ConditionalEffect {
    pub condition: Condition,
    pub effect: Effect,
}

impl ConditionalEffect {
    pub fn new(condition: Condition, effect: Effect) -> Self {
        Self { condition, effect }
    }

    pub fn dump(&self, indent: &str) {
        println!("{}if", indent);
        // TODO: Implement condition dump
        println!("{}  (condition)", indent);
        println!("{}then", indent);
        self.effect.dump();
    }

    pub fn normalize(&self) -> NormalizedEffect {
        // TODO: Implement normalization
        NormalizedEffect::Single(self.effect.clone())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum NormalizedEffect {
    Single(Effect),
    Conjunctive(Vec<Effect>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct UniversalEffect {
    pub parameters: Vec<TypedObject>,
    pub effect: Box<Effect>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FunctionAssignment {
    pub fluent: FunctionExpression,
    pub expression: FunctionExpression,
    pub assign_type: AssignType,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AssignType {
    Assign,
    Increase,
    Decrease,
    ScaleUp,
    ScaleDown,
}
