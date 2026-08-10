/// Effect types for PDDL actions.
use std::fmt;

use super::conditions::{Condition, Conjunction};
use super::f_expression::FunctionAssignment;
use super::pddl_types::TypedObject;

/// An effect consists of parameters (for universal effects), a condition, and a primitive effect.
#[derive(Debug, Clone)]
pub struct Effect {
    pub parameters: Vec<TypedObject>,
    pub condition: Condition,
    /// The literal (add or delete) effect
    pub peffect: Condition, // Always Atom or NegatedAtom
}

impl Effect {
    pub fn new(parameters: Vec<TypedObject>, condition: Condition, peffect: Condition) -> Self {
        Effect {
            parameters,
            condition,
            peffect,
        }
    }
}

impl fmt::Display for Effect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Effect({:?}, {}, {})",
            self.parameters, self.condition, self.peffect
        )
    }
}

#[derive(Debug, Clone)]
pub struct ConditionalEffect {
    pub condition: Condition,
    pub effect: Box<EffectType>,
}

impl ConditionalEffect {
    pub fn new(condition: Condition, effect: EffectType) -> Self {
        ConditionalEffect {
            condition,
            effect: Box::new(effect),
        }
    }
}

#[derive(Debug, Clone)]
pub struct UniversalEffect {
    pub parameters: Vec<TypedObject>,
    pub effect: Box<EffectType>,
}

impl UniversalEffect {
    pub fn new(parameters: Vec<TypedObject>, effect: EffectType) -> Self {
        UniversalEffect {
            parameters,
            effect: Box::new(effect),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ConjunctiveEffect {
    pub effects: Vec<EffectType>,
}

impl ConjunctiveEffect {
    pub fn new(effects: Vec<EffectType>) -> Self {
        ConjunctiveEffect { effects }
    }
}

#[derive(Debug, Clone)]
pub struct SimpleEffect {
    pub effect: Condition, // Always an Atom or NegatedAtom
}

impl SimpleEffect {
    pub fn new(effect: Condition) -> Self {
        SimpleEffect { effect }
    }
}

#[derive(Debug, Clone)]
pub struct NumericEffect {
    pub effect: FunctionAssignment,
}

impl NumericEffect {
    pub fn new(effect: FunctionAssignment) -> Self {
        NumericEffect { effect }
    }
}

/// Combined effect type enum for the parser output
/// (before normalization into Effect structs)
#[derive(Debug, Clone)]
pub enum EffectType {
    Simple(SimpleEffect),
    Numeric(NumericEffect),
    Conditional(ConditionalEffect),
    Universal(UniversalEffect),
    Conjunctive(ConjunctiveEffect),
}

impl EffectType {
    /// Converts the effect tree into a flat list of Effect structs.
    pub fn normalize(&self) -> Vec<(Vec<TypedObject>, Condition, EffectKind)> {
        self.normalize_aux(vec![], Condition::Truth)
    }

    fn normalize_aux(
        &self,
        params: Vec<TypedObject>,
        condition: Condition,
    ) -> Vec<(Vec<TypedObject>, Condition, EffectKind)> {
        match self {
            EffectType::Simple(se) => {
                vec![(params, condition, EffectKind::Literal(se.effect.clone()))]
            }
            EffectType::Numeric(ne) => {
                vec![(params, condition, EffectKind::Numeric(ne.effect.clone()))]
            }
            EffectType::Conditional(ce) => {
                let new_condition = match condition {
                    Condition::Truth => ce.condition.clone(),
                    _ => Condition::Conjunction(Conjunction::new(vec![
                        condition,
                        ce.condition.clone(),
                    ])),
                };
                ce.effect.normalize_aux(params, new_condition)
            }
            EffectType::Universal(ue) => {
                let mut new_params = params;
                new_params.extend(ue.parameters.clone());
                ue.effect.normalize_aux(new_params, condition)
            }
            EffectType::Conjunctive(ce) => {
                let mut result = vec![];
                for eff in &ce.effects {
                    result.extend(eff.normalize_aux(params.clone(), condition.clone()));
                }
                result
            }
        }
    }

    /// Extracts the cost effect from a conjunctive effect and returns the remaining effects + cost.
    pub fn extract_cost(&self) -> (EffectType, Option<FunctionAssignment>) {
        match self {
            EffectType::Conjunctive(ce) => {
                let mut new_effects = vec![];
                let mut cost_effect = None;
                for eff in &ce.effects {
                    match eff {
                        EffectType::Numeric(ne) if ne.effect.is_cost_assignment() => {
                            cost_effect = Some(ne.effect.clone());
                            new_effects.push(eff.clone());
                        }
                        _ => {
                            new_effects.push(eff.clone());
                        }
                    }
                }
                if new_effects.len() == 1 {
                    (new_effects.into_iter().next().unwrap(), cost_effect)
                } else {
                    (
                        EffectType::Conjunctive(ConjunctiveEffect::new(new_effects)),
                        cost_effect,
                    )
                }
            }
            _ => (self.clone(), None),
        }
    }
}

/// Distinguishes literal effects from numeric effects after normalization
#[derive(Debug, Clone)]
pub enum EffectKind {
    Literal(Condition),
    Numeric(FunctionAssignment),
}
