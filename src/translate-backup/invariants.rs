use crate::translate::constraints::ConstraintSystem;
use crate::translate::pddl::Condition;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Literal {
    pub predicate: String,
    pub args: Vec<String>,
    pub negated: bool,
}

impl Literal {
    pub fn negate(&self) -> Self {
        Self {
            predicate: self.predicate.clone(),
            args: self.args.clone(),
            negated: !self.negated,
        }
    }
}

#[derive(Clone, Debug)]
pub struct EffectView {
    pub parameters: Vec<String>,
    pub condition: Condition,
    pub peffect: Literal,
}

#[derive(Clone, Debug)]
pub struct ActionView {
    pub name: String,
    pub parameters: Vec<String>,
    pub precondition: Condition,
    pub effects: Vec<EffectView>,
}

#[derive(Clone, Debug)]
pub struct InvariantPart {
    pub predicate: String,
    pub order: Vec<usize>,
    pub omitted_pos: i32,
}

impl PartialEq for InvariantPart {
    fn eq(&self, other: &Self) -> bool {
        self.predicate == other.predicate && self.order == other.order
    }
}

impl Eq for InvariantPart {}

impl Hash for InvariantPart {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.predicate.hash(state);
        self.order.hash(state);
    }
}

impl InvariantPart {
    pub fn new(predicate: String, order: Vec<usize>, omitted_pos: i32) -> Self {
        Self {
            predicate,
            order,
            omitted_pos,
        }
    }
    pub fn arity(&self) -> usize {
        self.order.len()
    }
    pub fn get_parameters(&self, atom: &Literal) -> Vec<String> {
        self.order.iter().map(|&p| atom.args[p].clone()).collect()
    }
    pub fn instantiate(&self, parameters: &[String]) -> String {
        let mut args =
            vec!["?X".to_string(); self.order.len() + if self.omitted_pos != -1 { 1 } else { 0 }];
        for (arg, &pos) in parameters.iter().zip(self.order.iter()) {
            args[pos] = arg.clone();
        }
        format!("{}({})", self.predicate, args.join(", "))
    }
    pub fn get_assignment(
        &self,
        parameters: &[String],
        literal: &Literal,
    ) -> crate::translate::constraints::Assignment {
        // Build equalities: [(param_name, literal_arg_at_position), ...]
        let mut equalities: Vec<(String, String)> = Vec::new();
        for (arg, &pos) in parameters.iter().zip(self.order.iter()) {
            let lit_arg = literal.args[pos].clone();
            equalities.push((arg.clone(), lit_arg));
        }
        crate::translate::constraints::Assignment::new(equalities)
    }

    fn possible_mappings(
        &self,
        own_literal: &Literal,
        other_literal: &Literal,
    ) -> Vec<Vec<(usize, i32)>> {
        let mut allowed_omissions = other_literal.args.len() as i32 - self.order.len() as i32;
        if allowed_omissions != 0 && allowed_omissions != 1 {
            return Vec::new();
        }
        let own_parameters = self.get_parameters(own_literal);
        let arg_to_ordered_pos = invert_list(&own_parameters);
        let other_arg_to_pos = invert_list(&other_literal.args);
        let mut factored_mapping: Vec<(Vec<usize>, Vec<i32>)> = Vec::new();

        for (key, other_positions) in other_arg_to_pos {
            let mut own_positions: Vec<i32> = arg_to_ordered_pos
                .get(&key)
                .cloned()
                .unwrap_or_default()
                .into_iter()
                .map(|p| p as i32)
                .collect();
            let len_diff = own_positions.len() as i32 - other_positions.len() as i32;
            if len_diff >= 1 || len_diff <= -2 || (len_diff == -1 && allowed_omissions == 0) {
                return Vec::new();
            }
            if len_diff == -1 {
                own_positions.push(-1);
                allowed_omissions = 0;
            }
            factored_mapping.push((other_positions, own_positions));
        }
        instantiate_factored_mapping(&factored_mapping)
    }

    pub fn possible_matches(&self, own_literal: &Literal, other_literal: &Literal) -> Vec<Self> {
        let mut result = Vec::new();
        for mapping in self.possible_mappings(own_literal, other_literal) {
            let mut new_order: Vec<Option<usize>> = vec![None; self.order.len()];
            let mut omitted: i32 = -1;
            for (key, value) in mapping {
                if value == -1 {
                    omitted = key as i32;
                } else {
                    let idx = value as usize;
                    if idx < new_order.len() {
                        new_order[idx] = Some(key);
                    }
                }
            }
            if new_order.iter().any(|v| v.is_none()) {
                continue;
            }
            let order: Vec<usize> = new_order.into_iter().map(|v| v.unwrap()).collect();
            result.push(InvariantPart::new(
                other_literal.predicate.clone(),
                order,
                omitted,
            ));
        }
        result
    }

    // Rough equality check used in some refinement heuristics: compare parameters
    pub fn matches(
        &self,
        other: &InvariantPart,
        own_literal: &Literal,
        other_literal: &Literal,
    ) -> bool {
        self.get_parameters(own_literal) == other.get_parameters(other_literal)
    }
}

#[derive(Clone, Debug)]
pub struct Invariant {
    pub parts: Vec<InvariantPart>,
    pub predicates: HashSet<String>,
    pub predicate_to_part: HashMap<String, InvariantPart>,
}

impl Invariant {
    pub fn new(parts: Vec<InvariantPart>) -> Self {
        let mut preds = HashSet::new();
        let mut predicate_to_part = HashMap::new();
        for p in &parts {
            preds.insert(p.predicate.clone());
            predicate_to_part.insert(p.predicate.clone(), p.clone());
        }
        Invariant {
            parts,
            predicates: preds,
            predicate_to_part,
        }
    }
    pub fn arity(&self) -> usize {
        self.parts.first().map(|p| p.arity()).unwrap_or(0)
    }
    pub fn get_parameters_for_atom(&self, atom: &Literal) -> Vec<String> {
        if let Some(part) = self.predicate_to_part.get(&atom.predicate) {
            return part.get_parameters(atom);
        }
        Vec::new()
    }
    pub fn get_covering_assignments(
        &self,
        parameters: &[String],
        atom: &Literal,
    ) -> Vec<crate::translate::constraints::Assignment> {
        if let Some(part) = self.predicate_to_part.get(&atom.predicate) {
            return vec![part.get_assignment(parameters, atom)];
        }
        Vec::new()
    }
    pub fn instantiate(&self, parameters: &[String]) -> Vec<String> {
        let mut parts = self.parts.clone();
        parts.sort_by(|a, b| {
            let pc = a.predicate.cmp(&b.predicate);
            if pc != std::cmp::Ordering::Equal {
                pc
            } else {
                a.order.cmp(&b.order)
            }
        });
        parts
            .iter()
            .map(|part| part.instantiate(parameters))
            .collect()
    }
    pub fn with_added_part(&self, part: InvariantPart) -> Invariant {
        let mut parts = self.parts.clone();
        parts.push(part);
        Invariant::new(parts)
    }
}
impl PartialEq for Invariant {
    fn eq(&self, other: &Self) -> bool {
        // Compare parts as sets: same parts implies equality
        let mut a = self.parts.clone();
        let mut b = other.parts.clone();
        a.sort_by(|x, y| {
            let pc = x.predicate.cmp(&y.predicate);
            if pc != std::cmp::Ordering::Equal {
                pc
            } else {
                x.order.cmp(&y.order)
            }
        });
        b.sort_by(|x, y| {
            let pc = x.predicate.cmp(&y.predicate);
            if pc != std::cmp::Ordering::Equal {
                pc
            } else {
                x.order.cmp(&y.order)
            }
        });
        a == b
    }
}
impl Eq for Invariant {}
impl Hash for Invariant {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Hash parts sorted by predicate to get deterministic hash
        let mut parts = self.parts.clone();
        parts.sort_by(|x, y| {
            let pc = x.predicate.cmp(&y.predicate);
            if pc != std::cmp::Ordering::Equal {
                pc
            } else {
                x.order.cmp(&y.order)
            }
        });
        for p in parts {
            p.hash(state);
        }
    }
}

// helper: extract literals from a Condition (literal or conjunction)
pub fn get_literals(cond: &Condition) -> Vec<Literal> {
    match cond {
        Condition::Atom(pred, args) => vec![Literal {
            predicate: pred.clone(),
            args: args.clone(),
            negated: false,
        }],
        Condition::Not(inner) => match inner.as_ref() {
            Condition::Atom(pred, args) => vec![Literal {
                predicate: pred.clone(),
                args: args.clone(),
                negated: true,
            }],
            _ => Vec::new(),
        },
        Condition::And(v) => v.iter().flat_map(get_literals).collect(),
        _ => Vec::new(),
    }
}

pub fn ensure_conjunction_sat(system: &mut ConstraintSystem, parts: &[Vec<Literal>]) {
    let mut pos: HashMap<String, HashSet<Literal>> = HashMap::new();
    let mut neg: HashMap<String, HashSet<Literal>> = HashMap::new();

    for literal in parts.iter().flatten() {
        if literal.predicate == "=" {
            if literal.args.len() >= 2 {
                if literal.negated {
                    let n = crate::translate::constraints::NegativeClause::new(vec![(
                        literal.args[0].clone(),
                        literal.args[1].clone(),
                    )]);
                    system.add_negative_clause(n);
                } else {
                    let a = crate::translate::constraints::Assignment::new(vec![(
                        literal.args[0].clone(),
                        literal.args[1].clone(),
                    )]);
                    system.add_assignment_disjunction(vec![a]);
                }
            }
            continue;
        }
        if literal.negated {
            neg.entry(literal.predicate.clone())
                .or_default()
                .insert(literal.clone());
        } else {
            pos.entry(literal.predicate.clone())
                .or_default()
                .insert(literal.clone());
        }
    }

    for (pred, posatoms) in pos.iter() {
        if let Some(negatoms) = neg.get(pred) {
            for posatom in posatoms {
                for negatom in negatoms {
                    let parts: Vec<(String, String)> = negatom
                        .args
                        .iter()
                        .zip(posatom.args.iter())
                        .map(|(a, b)| (a.clone(), b.clone()))
                        .collect();
                    if !parts.is_empty() {
                        system.add_negative_clause(
                            crate::translate::constraints::NegativeClause::new(parts),
                        );
                    }
                }
            }
        }
    }
}

pub fn ensure_cover(
    system: &mut ConstraintSystem,
    literal: &Literal,
    invariant: &Invariant,
    inv_vars: &[String],
) {
    // Convert to assignment(s) and add to the system.
    let assignments = invariant.get_covering_assignments(inv_vars, literal);
    for a in assignments {
        system.add_assignment_disjunction(vec![a]);
    }
}

pub fn ensure_inequality(system: &mut ConstraintSystem, lit1: &Literal, lit2: &Literal) {
    // If both are atoms and have parts, add a NegativeClause with paired positions
    if lit1.predicate == lit2.predicate && !lit1.args.is_empty() {
        let mut parts: Vec<(String, String)> = Vec::new();
        let len = std::cmp::min(lit1.args.len(), lit2.args.len());
        for i in 0..len {
            parts.push((lit1.args[i].clone(), lit2.args[i].clone()));
        }
        if !parts.is_empty() {
            system.add_negative_clause(crate::translate::constraints::NegativeClause::new(parts));
        }
    }
}

fn permutations(items: &[i32]) -> Vec<Vec<i32>> {
    if items.is_empty() {
        return vec![vec![]];
    }
    let mut result = Vec::new();
    for i in 0..items.len() {
        let mut rest = Vec::new();
        for (idx, item) in items.iter().enumerate() {
            if idx != i {
                rest.push(*item);
            }
        }
        for mut perm in permutations(&rest) {
            let mut new_perm = vec![items[i]];
            new_perm.append(&mut perm);
            result.push(new_perm);
        }
    }
    result
}

fn instantiate_factored_mapping(pairs: &[(Vec<usize>, Vec<i32>)]) -> Vec<Vec<(usize, i32)>> {
    if pairs.is_empty() {
        return vec![Vec::new()];
    }
    let mut part_mappings: Vec<Vec<Vec<(usize, i32)>>> = Vec::new();
    for (preimg, img) in pairs {
        let mut mappings_for_pair = Vec::new();
        for perm in permutations(img) {
            let zipped: Vec<(usize, i32)> = preimg.iter().cloned().zip(perm.into_iter()).collect();
            mappings_for_pair.push(zipped);
        }
        part_mappings.push(mappings_for_pair);
    }

    let mut result: Vec<Vec<(usize, i32)>> = vec![Vec::new()];
    for part in part_mappings {
        let mut next = Vec::new();
        for base in &result {
            for item in &part {
                let mut combined = base.clone();
                combined.extend(item.iter().cloned());
                next.push(combined);
            }
        }
        result = next;
    }
    result
}

pub fn invert_list(list: &[String]) -> HashMap<String, Vec<usize>> {
    let mut result: HashMap<String, Vec<usize>> = HashMap::new();
    for (pos, arg) in list.iter().enumerate() {
        result.entry(arg.clone()).or_default().push(pos);
    }
    result
}

pub fn find_unique_variables(action: &ActionView, invariant: &Invariant) -> Vec<String> {
    let mut params: HashSet<String> = HashSet::new();
    for p in &action.parameters {
        params.insert(p.clone());
    }
    for eff in &action.effects {
        for p in &eff.parameters {
            params.insert(p.clone());
        }
    }
    let mut inv_vars = Vec::new();
    let mut counter = 0usize;
    for _ in 0..invariant.arity() {
        loop {
            let candidate = format!("?v{}", counter);
            counter += 1;
            if !params.contains(&candidate) {
                inv_vars.push(candidate);
                break;
            }
        }
    }
    inv_vars
}

impl Invariant {
    pub fn operator_too_heavy(&self, action: &ActionView) -> bool {
        let add_effects: Vec<&EffectView> = action
            .effects
            .iter()
            .filter(|eff| {
                !eff.peffect.negated && self.predicate_to_part.contains_key(&eff.peffect.predicate)
            })
            .collect();

        if add_effects.len() <= 1 {
            return false;
        }

        let inv_vars = find_unique_variables(action, self);
        for i in 0..add_effects.len() {
            for j in (i + 1)..add_effects.len() {
                let eff1 = add_effects[i];
                let eff2 = add_effects[j];
                let mut system = ConstraintSystem::new();
                ensure_inequality(&mut system, &eff1.peffect, &eff2.peffect);
                ensure_cover(&mut system, &eff1.peffect, self, &inv_vars);
                ensure_cover(&mut system, &eff2.peffect, self, &inv_vars);
                let parts = vec![
                    get_literals(&action.precondition),
                    get_literals(&eff1.condition),
                    get_literals(&eff2.condition),
                    vec![eff1.peffect.negate()],
                    vec![eff2.peffect.negate()],
                ];
                ensure_conjunction_sat(&mut system, &parts);
                if system.is_solvable() {
                    return true;
                }
            }
        }
        false
    }

    pub fn operator_unbalanced(
        &self,
        action: &ActionView,
        enqueue_func: &mut impl FnMut(Invariant),
    ) -> bool {
        let inv_vars = find_unique_variables(action, self);
        let relevant_effects: Vec<&EffectView> = action
            .effects
            .iter()
            .filter(|eff| self.predicate_to_part.contains_key(&eff.peffect.predicate))
            .collect();
        let add_effects: Vec<&EffectView> = relevant_effects
            .iter()
            .cloned()
            .filter(|eff| !eff.peffect.negated)
            .collect();
        let del_effects: Vec<&EffectView> = relevant_effects
            .iter()
            .cloned()
            .filter(|eff| eff.peffect.negated)
            .collect();

        for eff in add_effects {
            if self.add_effect_unbalanced(action, eff, &del_effects, &inv_vars, enqueue_func) {
                return true;
            }
        }
        false
    }

    pub fn minimal_covering_renamings(
        &self,
        action: &ActionView,
        add_effect: &EffectView,
        inv_vars: &[String],
    ) -> Vec<ConstraintSystem> {
        let assigs = self.get_covering_assignments(inv_vars, &add_effect.peffect);
        let mut minimal_renamings = Vec::new();
        let params = action.parameters.clone();

        for assignment in assigs {
            let mut assignment_copy = assignment.clone();
            let mapping = assignment_copy.get_mapping().unwrap_or_default();
            let mut system = ConstraintSystem::new();
            system.add_assignment(assignment);
            if params.len() > 1 {
                for i in 0..params.len() {
                    for j in (i + 1)..params.len() {
                        let n1 = &params[i];
                        let n2 = &params[j];
                        let m1 = mapping.get(n1).cloned().unwrap_or_else(|| n1.clone());
                        let m2 = mapping.get(n2).cloned().unwrap_or_else(|| n2.clone());
                        if m1 != m2 {
                            let negative_clause =
                                crate::translate::constraints::NegativeClause::new(vec![(
                                    n1.clone(),
                                    n2.clone(),
                                )]);
                            system.add_negative_clause(negative_clause);
                        }
                    }
                }
            }
            minimal_renamings.push(system);
        }
        minimal_renamings
    }

    pub fn add_effect_unbalanced(
        &self,
        action: &ActionView,
        add_effect: &EffectView,
        del_effects: &[&EffectView],
        inv_vars: &[String],
        enqueue_func: &mut impl FnMut(Invariant),
    ) -> bool {
        let mut minimal_renamings = self.minimal_covering_renamings(action, add_effect, inv_vars);

        let mut lhs_by_pred: HashMap<String, Vec<Literal>> = HashMap::new();
        let mut lhs_literals = Vec::new();
        lhs_literals.extend(get_literals(&action.precondition));
        lhs_literals.extend(get_literals(&add_effect.condition));
        lhs_literals.push(add_effect.peffect.negate());
        for lit in lhs_literals {
            lhs_by_pred
                .entry(lit.predicate.clone())
                .or_default()
                .push(lit);
        }

        for del_effect in del_effects {
            minimal_renamings = self.unbalanced_renamings(
                del_effect,
                add_effect,
                inv_vars,
                &lhs_by_pred,
                &minimal_renamings,
            );
            if minimal_renamings.is_empty() {
                return false;
            }
        }

        self.refine_candidate(add_effect, action, enqueue_func);
        true
    }

    pub fn refine_candidate(
        &self,
        add_effect: &EffectView,
        action: &ActionView,
        enqueue_func: &mut impl FnMut(Invariant),
    ) {
        let part = match self.predicate_to_part.get(&add_effect.peffect.predicate) {
            Some(p) => p,
            None => return,
        };
        for del_eff in action.effects.iter().filter(|eff| eff.peffect.negated) {
            if self
                .predicate_to_part
                .contains_key(&del_eff.peffect.predicate)
            {
                continue;
            }
            for matched in part.possible_matches(&add_effect.peffect, &del_eff.peffect) {
                enqueue_func(self.with_added_part(matched));
            }
        }
    }

    pub fn unbalanced_renamings(
        &self,
        del_effect: &EffectView,
        add_effect: &EffectView,
        inv_vars: &[String],
        lhs_by_pred: &HashMap<String, Vec<Literal>>,
        unbalanced_renamings: &[ConstraintSystem],
    ) -> Vec<ConstraintSystem> {
        let mut system = ConstraintSystem::new();
        ensure_inequality(&mut system, &add_effect.peffect, &del_effect.peffect);
        ensure_cover(&mut system, &del_effect.peffect, self, inv_vars);

        let mut check_constants = false;
        let mut constant_test_system = ConstraintSystem::new();
        if let Some(first_disj) = system.combinatorial_assignments.first() {
            if let Some(first_assignment) = first_disj.first() {
                for (a, b) in &first_assignment.equalities {
                    if !b.starts_with('?') {
                        check_constants = true;
                        let neg_clause =
                            crate::translate::constraints::NegativeClause::new(vec![(
                                a.clone(),
                                b.clone(),
                            )]);
                        constant_test_system.add_negative_clause(neg_clause);
                    }
                }
            }
        }

        ensure_inequality(&mut system, &add_effect.peffect, &del_effect.peffect);

        let mut still_unbalanced = Vec::new();
        for renaming in unbalanced_renamings {
            if check_constants {
                let new_sys = constant_test_system.combine(renaming);
                if new_sys.is_solvable() {
                    still_unbalanced.push(renaming.clone());
                    continue;
                }
            }
            let mut new_sys = system.combine(renaming);
            if self.lhs_satisfiable(renaming, lhs_by_pred) {
                let implies_system = self.imply_del_effect(del_effect, lhs_by_pred);
                if implies_system.is_none() {
                    still_unbalanced.push(renaming.clone());
                    continue;
                }
                new_sys = new_sys.combine(&implies_system.unwrap());
            }
            if !new_sys.is_solvable() {
                still_unbalanced.push(renaming.clone());
            }
        }
        still_unbalanced
    }

    pub fn lhs_satisfiable(
        &self,
        renaming: &ConstraintSystem,
        lhs_by_pred: &HashMap<String, Vec<Literal>>,
    ) -> bool {
        let mut system = renaming.copy();
        let parts: Vec<Vec<Literal>> = lhs_by_pred.values().cloned().collect();
        ensure_conjunction_sat(&mut system, &parts);
        system.is_solvable()
    }

    pub fn imply_del_effect(
        &self,
        del_effect: &EffectView,
        lhs_by_pred: &HashMap<String, Vec<Literal>>,
    ) -> Option<ConstraintSystem> {
        let mut implies_system = ConstraintSystem::new();
        let mut literals = get_literals(&del_effect.condition);
        literals.push(del_effect.peffect.negate());

        for literal in literals {
            let mut poss_assignments = Vec::new();
            if let Some(matches) = lhs_by_pred.get(&literal.predicate) {
                for m in matches {
                    if m.negated != literal.negated {
                        continue;
                    }
                    let equalities: Vec<(String, String)> = literal
                        .args
                        .iter()
                        .cloned()
                        .zip(m.args.iter().cloned())
                        .collect();
                    let a = crate::translate::constraints::Assignment::new(equalities);
                    poss_assignments.push(a);
                }
            }
            if poss_assignments.is_empty() {
                return None;
            }
            implies_system.add_assignment_disjunction(poss_assignments);
        }
        Some(implies_system)
    }
}
