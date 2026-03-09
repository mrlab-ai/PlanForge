/// Port of axiom_rules.py
/// Handles axiom layers, simplification, and negative axiom computation.

use std::collections::{HashMap, HashSet};

use super::pddl::conditions::*;
use super::pddl::axioms::PropositionalAxiom;
use super::pddl::actions::PropositionalAction;

/// Python: def handle_axioms(operators, axioms, goal_list, global_constraint)
/// Returns (processed_axioms, axiom_init_atoms, axiom_layer_dict)
pub fn handle_axioms(
    operators: &[PropositionalAction],
    axioms: Vec<PropositionalAxiom>,
    goal_list: &[Condition],
    global_constraint: &Condition,
) -> (Vec<PropositionalAxiom>, Vec<Atom>, HashMap<Atom, i32>) {
    if axioms.is_empty() {
        return (vec![], vec![], HashMap::new());
    }

    let mut axioms_by_atom = get_axioms_by_atom(&axioms);
    let axiom_literals = compute_necessary_axiom_literals(
        &axioms_by_atom,
        operators,
        goal_list,
        global_constraint,
    );
    let axiom_init = compute_axiom_init(&axioms_by_atom, &axiom_literals);
    let _simplified_axioms = simplify_axioms(&mut axioms_by_atom, &axiom_literals);
    let processed_axioms = compute_negative_axioms(&axioms_by_atom, &axiom_literals);
    let axiom_layers = compute_axiom_layers(&processed_axioms, &axiom_init);

    (processed_axioms, axiom_init.into_iter().collect(), axiom_layers)
}

fn get_axioms_by_atom(axioms: &[PropositionalAxiom]) -> HashMap<Atom, Vec<PropositionalAxiom>> {
    let mut axioms_by_atom: HashMap<Atom, Vec<PropositionalAxiom>> = HashMap::new();
    for axiom in axioms {
        if let Some(effect_atom) = axiom.effect.literal_positive() {
            axioms_by_atom
                .entry(effect_atom)
                .or_default()
                .push(axiom.clone());
        }
    }
    axioms_by_atom
}

fn compute_necessary_axiom_literals(
    axioms_by_atom: &HashMap<Atom, Vec<PropositionalAxiom>>,
    operators: &[PropositionalAction],
    goal_list: &[Condition],
    global_constraint: &Condition,
) -> HashSet<Condition> {
    let mut necessary_literals: HashSet<Condition> = HashSet::new();
    let mut queue: Vec<Condition> = vec![];

    let register_literals = |literals: &[Condition],
                             negated: bool,
                             necessary_literals: &mut HashSet<Condition>,
                             queue: &mut Vec<Condition>| {
        for literal in literals {
            if let Some(positive_atom) = literal.literal_positive() {
                if axioms_by_atom.contains_key(&positive_atom) {
                    let normalized = if negated {
                        literal.negate_literal().unwrap_or_else(|| literal.clone())
                    } else {
                        literal.clone()
                    };
                    if necessary_literals.insert(normalized.clone()) {
                        queue.push(normalized);
                    }
                }
            }
        }
    };

    register_literals(goal_list, false, &mut necessary_literals, &mut queue);
    register_literals(
        std::slice::from_ref(global_constraint),
        false,
        &mut necessary_literals,
        &mut queue,
    );

    for operator in operators {
        register_literals(
            &operator.precondition,
            false,
            &mut necessary_literals,
            &mut queue,
        );
        for (condition, _) in &operator.add_effects {
            register_literals(condition, false, &mut necessary_literals, &mut queue);
        }
        for (condition, _) in &operator.del_effects {
            register_literals(condition, true, &mut necessary_literals, &mut queue);
        }
    }

    while let Some(literal) = queue.pop() {
        if let Some(positive_atom) = literal.literal_positive() {
            if let Some(axioms) = axioms_by_atom.get(&positive_atom) {
                for axiom in axioms {
                    register_literals(
                        &axiom.condition,
                        literal.is_negated(),
                        &mut necessary_literals,
                        &mut queue,
                    );
                }
            }
        }
    }

    necessary_literals
}

fn compute_axiom_init(
    axioms_by_atom: &HashMap<Atom, Vec<PropositionalAxiom>>,
    necessary_literals: &HashSet<Condition>,
) -> HashSet<Atom> {
    let mut result = HashSet::new();
    for atom in axioms_by_atom.keys() {
        let positive = Condition::Atom(atom.clone());
        let negative = Condition::NegatedAtom(atom.negate());
        if !necessary_literals.contains(&positive) && necessary_literals.contains(&negative) {
            result.insert(atom.clone());
        }
    }
    result
}

fn simplify_axioms(
    axioms_by_atom: &mut HashMap<Atom, Vec<PropositionalAxiom>>,
    necessary_literals: &HashSet<Condition>,
) -> Vec<PropositionalAxiom> {
    let necessary_atoms: HashSet<Atom> = necessary_literals
        .iter()
        .filter_map(|literal| literal.literal_positive())
        .collect();

    let mut new_axioms = vec![];
    for atom in necessary_atoms {
        if let Some(axioms) = axioms_by_atom.get(&atom).cloned() {
            let simplified = simplify(axioms);
            axioms_by_atom.insert(atom, simplified.clone());
            new_axioms.extend(simplified);
        }
    }
    new_axioms
}

fn simplify(mut axioms: Vec<PropositionalAxiom>) -> Vec<PropositionalAxiom> {
    for axiom in &mut axioms {
        axiom.condition
            .sort_by_key(|condition| format!("{:?}", condition));
        remove_duplicate_conditions(&mut axiom.condition);
    }

    let mut axioms_by_literal: HashMap<Condition, HashSet<usize>> = HashMap::new();
    for (index, axiom) in axioms.iter().enumerate() {
        for literal in &axiom.condition {
            axioms_by_literal
                .entry(literal.clone())
                .or_default()
                .insert(index);
        }
    }

    let mut axioms_to_skip: HashSet<usize> = HashSet::new();
    for (index, axiom) in axioms.iter().enumerate() {
        if axioms_to_skip.contains(&index) {
            continue;
        }
        if axiom.condition.is_empty() {
            return vec![axiom.clone()];
        }

        let mut literals = axiom.condition.iter();
        let first_literal = literals.next().unwrap();
        let mut dominated_axioms = axioms_by_literal
            .get(first_literal)
            .cloned()
            .unwrap_or_default();
        for literal in literals {
            if let Some(candidates) = axioms_by_literal.get(literal) {
                dominated_axioms = dominated_axioms
                    .intersection(candidates)
                    .copied()
                    .collect();
            } else {
                dominated_axioms.clear();
                break;
            }
        }
        for dominated_axiom in dominated_axioms {
            if dominated_axiom != index {
                axioms_to_skip.insert(dominated_axiom);
            }
        }
    }

    axioms
        .into_iter()
        .enumerate()
        .filter_map(|(index, axiom)| (!axioms_to_skip.contains(&index)).then_some(axiom))
        .collect()
}

fn remove_duplicate_conditions(conditions: &mut Vec<Condition>) {
    conditions.dedup();
}

fn compute_negative_axioms(
    axioms_by_atom: &HashMap<Atom, Vec<PropositionalAxiom>>,
    necessary_literals: &HashSet<Condition>,
) -> Vec<PropositionalAxiom> {
    let mut new_axioms = vec![];
    let mut literals: Vec<Condition> = necessary_literals.iter().cloned().collect();
    literals.sort_by_key(|literal| format!("{:?}", literal));
    for literal in literals {
        if literal.is_negated() {
            if let Some(atom) = literal.literal_positive() {
                if let Some(axioms) = axioms_by_atom.get(&atom) {
                    new_axioms.extend(negate(axioms));
                }
            }
        } else if let Some(atom) = literal.literal_positive() {
            if let Some(axioms) = axioms_by_atom.get(&atom) {
                new_axioms.extend(axioms.clone());
            }
        }
    }
    new_axioms
}

pub fn negate(axioms: &[PropositionalAxiom]) -> Vec<PropositionalAxiom> {
    assert!(!axioms.is_empty());

    let initial_effect = axioms[0]
        .effect
        .negate_literal()
        .unwrap_or_else(|| axioms[0].effect.clone());
    let mut result = vec![PropositionalAxiom::new(
        axioms[0].name.clone(),
        vec![],
        initial_effect,
    )];

    for axiom in axioms {
        let condition = &axiom.condition;
        if condition.is_empty() {
            return vec![];
        } else if condition.len() == 1 {
            let new_literal = condition[0]
                .negate_literal()
                .unwrap_or_else(|| condition[0].clone());
            for result_axiom in &mut result {
                result_axiom.condition.push(new_literal.clone());
            }
        } else {
            let mut new_result = vec![];
            for literal in condition {
                let negated_literal = literal
                    .negate_literal()
                    .unwrap_or_else(|| literal.clone());
                for result_axiom in &result {
                    let mut new_axiom = result_axiom.clone_axiom();
                    new_axiom.condition.push(negated_literal.clone());
                    new_result.push(new_axiom);
                }
            }
            result = new_result;
        }
    }

    simplify(result)
}

fn compute_axiom_layers(
    axioms: &[PropositionalAxiom],
    axiom_init: &HashSet<Atom>,
) -> HashMap<Atom, i32> {
    const NO_AXIOM: i32 = -1;
    const UNKNOWN_LAYER: i32 = -2;
    const FIRST_MARKER: i32 = -3;

    let mut depends_on: HashMap<Atom, HashSet<(Atom, i32)>> = HashMap::new();
    for axiom in axioms {
        let effect_atom = axiom.effect.literal_positive().unwrap();
        let effect_sign = !axiom.effect.is_negated();
        let effect_init_sign = axiom_init.contains(&effect_atom);
        if effect_sign != effect_init_sign {
            let entry = depends_on.entry(effect_atom.clone()).or_default();
            for condition in &axiom.condition {
                if let Some(condition_atom) = condition.literal_positive() {
                    let condition_sign = !condition.is_negated();
                    let condition_init_sign = axiom_init.contains(&condition_atom);
                    let bonus = if condition_sign == condition_init_sign { 1 } else { 0 };
                    entry.insert((condition_atom, bonus));
                }
            }
        }
    }

    let mut layers: HashMap<Atom, i32> = depends_on
        .keys()
        .cloned()
        .map(|atom| (atom, UNKNOWN_LAYER))
        .collect();

    fn find_level(
        atom: &Atom,
        marker: i32,
        depends_on: &HashMap<Atom, HashSet<(Atom, i32)>>,
        layers: &mut HashMap<Atom, i32>,
    ) -> i32 {
        let layer = *layers.get(atom).unwrap_or(&NO_AXIOM);
        if layer == NO_AXIOM {
            return 0;
        }
        if layer == marker {
            return 0;
        }
        if layer <= FIRST_MARKER {
            panic!("Cyclic dependencies in axioms; cannot stratify.");
        }
        if layer == UNKNOWN_LAYER {
            layers.insert(atom.clone(), marker);
            let mut new_layer = 0;
            if let Some(dependencies) = depends_on.get(atom) {
                for (condition_atom, bonus) in dependencies {
                    new_layer = new_layer.max(find_level(
                        condition_atom,
                        marker - *bonus,
                        depends_on,
                        layers,
                    ) + *bonus);
                }
            }
            layers.insert(atom.clone(), new_layer);
            return new_layer;
        }
        layer
    }

    let atoms: Vec<Atom> = depends_on.keys().cloned().collect();
    for atom in atoms {
        find_level(&atom, FIRST_MARKER, &depends_on, &mut layers);
    }

    layers
}
