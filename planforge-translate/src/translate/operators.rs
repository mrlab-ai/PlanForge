//! Grounded operators and axioms as SAS+ operators and axioms: their
//! conditions, effects and costs, with effect conditions that cannot matter
//! pruned.

use super::*;

// ============================================================
// translate_strips_operator
// ============================================================

pub(super) fn translate_strips_operator(
    translation: &mut Translation,
    operator: &PropositionalAction,
) -> Vec<SASOperator> {
    let conditions = translate_strips_conditions(translation, &operator.precondition);

    if conditions.is_none() {
        return vec![];
    }

    let mut sas_operators = vec![];
    for condition in conditions.unwrap() {
        if let Some(op) = translate_strips_operator_aux(translation, operator, &condition) {
            sas_operators.push(op);
        }
    }
    sas_operators
}

// ============================================================
// translate_strips_operator_aux
// ============================================================

pub(super) fn translate_strips_operator_aux(
    translation: &mut Translation,
    operator: &PropositionalAction,
    condition: &Assignment,
) -> Option<SASOperator> {
    // Collect all add effects
    let mut effects_by_variable: HashMap<usize, HashMap<usize, Vec<Assignment>>> = HashMap::new();
    let mut add_conds_by_variable: HashMap<usize, Vec<Vec<Condition>>> = HashMap::new();

    for (conditions_list, fact) in &operator.add_effects {
        let eff_condition_list = translate_strips_conditions(translation, conditions_list);
        if eff_condition_list.is_none() {
            continue; // Impossible condition
        }
        // An add effect of a reachable operator is a reachable atom, which
        // the dictionary encodes.
        let pairs = translation.dictionary.get(fact).unwrap_or_else(|| {
            panic!(
                "add effect {fact} of operator {} has no SAS encoding",
                operator.name
            )
        });
        for &(var, val) in pairs {
            effects_by_variable
                .entry(var)
                .or_default()
                .entry(val)
                .or_default()
                .extend(eff_condition_list.clone().unwrap());
            add_conds_by_variable
                .entry(var)
                .or_default()
                .push(conditions_list.clone());
        }
    }

    // Collect all del effects
    let mut del_effects_by_variable: HashMap<usize, HashMap<usize, Vec<Assignment>>> =
        HashMap::new();

    for (conditions_list, fact) in &operator.del_effects {
        let eff_condition_list = translate_strips_conditions(translation, conditions_list);
        if eff_condition_list.is_none() {
            continue;
        }
        // An atom without an encoding is never true, so deleting it does
        // nothing.
        if let Some(pairs) = translation.dictionary.get(fact) {
            for &(var, val) in pairs {
                del_effects_by_variable
                    .entry(var)
                    .or_default()
                    .entry(val)
                    .or_default()
                    .extend(eff_condition_list.clone().unwrap());
            }
        }
    }

    // Collect all (numeric) assignment effects
    let mut ass_effects_by_variable: HashMap<usize, HashMap<(String, usize), Vec<Assignment>>> =
        HashMap::new();

    for (conditions_list, assignment) in &operator.assign_effects {
        let eff_condition_list = translate_strips_conditions(translation, conditions_list);
        if eff_condition_list.is_none() {
            continue;
        }
        let expr_pne = assignment.expression.as_pne().unwrap_or_else(|| {
            panic!(
                "numeric effect of operator {} has non-PNE expression {}",
                operator.name, assignment.expression
            )
        });
        let &expr_var = translation.numeric.get(expr_pne).unwrap_or_else(|| {
            panic!(
                "numeric effect expression {expr_pne} of operator {} has no numeric variable",
                operator.name
            )
        });
        let &fluent_var = translation
            .numeric
            .get(&assignment.fluent)
            .unwrap_or_else(|| {
                panic!(
                    "numeric effect fluent {} of operator {} has no numeric variable",
                    assignment.fluent, operator.name
                )
            });
        ass_effects_by_variable
            .entry(fluent_var)
            .or_default()
            .entry((assignment.symbol.clone(), expr_var))
            .or_default()
            .extend(eff_condition_list.unwrap());
    }

    if let Some(cost_assignment) = &operator.cost {
        let expr_pne = cost_assignment.expression.as_pne().unwrap_or_else(|| {
            panic!(
                "cost effect of operator {} has non-PNE expression {}",
                operator.name, cost_assignment.expression
            )
        });
        let &expr_var = translation.numeric.get(expr_pne).unwrap_or_else(|| {
            panic!(
                "cost expression {expr_pne} of operator {} has no numeric variable",
                operator.name
            )
        });
        // `operator.cost` is bookkeeping on `total-cost`. A problem may optimize
        // another fluent instead, in which case `total-cost` is static and has no
        // SAS numeric variable (for example, the Depots fixture minimizes
        // `fuel-cost`). The scalar cost above still uses the expression, while no
        // live numeric effect is needed for the non-metric bookkeeping fluent.
        if let Some(&fluent_var) = translation.numeric.get(&cost_assignment.fluent) {
            ass_effects_by_variable
                .entry(fluent_var)
                .or_default()
                .entry((cost_assignment.symbol.clone(), expr_var))
                .or_default()
                .push(Assignment::default());
        }
    }

    // Handle del effects: add var=none_of_those when deleted and no add effect
    for (&var, del_vals) in &del_effects_by_variable {
        let add_conds = add_conds_by_variable
            .get(&var)
            .cloned()
            .unwrap_or_else(Vec::new);

        let no_add_effect_condition = negate_and_translate_condition(translation, &add_conds);

        if no_add_effect_condition.is_none() {
            continue; // Always an add effect
        }

        let none_of_those = translation.ranges[var] - 1;
        for (&val, conds) in del_vals {
            for cond in conds {
                let mut guard_cond = cond.clone();
                if guard_cond.get(var).is_some_and(|existing| existing != val) {
                    continue; // Condition inconsistent with deleted atom
                }
                guard_cond.set(var, val);

                for no_add_cond in no_add_effect_condition.as_ref().unwrap() {
                    let mut new_cond = guard_cond.clone();
                    let mut contradicts = false;
                    for (cvar, cval) in no_add_cond.iter() {
                        if new_cond.get(cvar).is_some_and(|existing| existing != cval) {
                            contradicts = true;
                            break;
                        }
                        new_cond.set(cvar, cval);
                    }
                    if !contradicts {
                        effects_by_variable
                            .entry(var)
                            .or_default()
                            .entry(none_of_those)
                            .or_default()
                            .push(new_cond);
                    }
                }
            }
        }
    }

    let cost = sas_operator_cost(translation, &operator.name, operator.cost.as_ref());
    build_sas_operator(
        translation,
        &operator.name,
        condition,
        &effects_by_variable,
        &ass_effects_by_variable,
        cost,
    )
}

/// Scalar cost of a SAS operator.
///
/// An action without an `(increase (total-cost) ...)` effect costs 1, PDDL's
/// default action cost. A cost effect is either a literal constant or — after
/// normalization, which rewrites every compound cost expression into a derived
/// function — a primitive numeric expression; the latter is evaluated in the
/// initial state, mirroring how Fast Downward resolves static cost functions.
/// A state-dependent cost is additionally recorded as an assignment effect on
/// the metric fluent by the caller, which is what the search actually uses for
/// tasks with a metric (`metric_operator_costs`).
///
/// Anything else cannot be turned into a number here; returning a default
/// would silently replace the action's real cost, so it is a hard error.
pub(super) fn sas_operator_cost(
    translation: &Translation,
    action_name: &str,
    cost: Option<&FunctionAssignment>,
) -> f64 {
    let (numeric_dictionary, num_vals) = (&translation.numeric, &translation.num_vals);
    let Some(cost) = cost else {
        return 1.0;
    };
    match &cost.expression {
        FunctionalExpression::NumericConstant(nc) => nc.value.value(),
        FunctionalExpression::PrimitiveNumericExpression(pne) => {
            let &var = numeric_dictionary.get(pne).unwrap_or_else(|| {
                panic!("cost expression {pne} of action {action_name} has no numeric variable")
            });
            num_vals[var].value()
        }
        other => panic!("action {action_name} has an unsupported cost expression {other}"),
    }
}

// ============================================================
// build_sas_operator
// ============================================================

pub(super) fn build_sas_operator(
    translation: &mut Translation,
    name: &str,
    condition: &Assignment,
    effects_by_variable: &HashMap<usize, HashMap<usize, Vec<Assignment>>>,
    ass_effects_by_variable: &HashMap<usize, HashMap<(String, usize), Vec<Assignment>>>,
    cost: f64,
) -> Option<SASOperator> {
    let Translation {
        ranges,
        relevant_numeric: relevant_numeric_variables,
        simplified_effect_conditions: simplified_effect_condition_counter,
        ..
    } = translation;
    let mut prevail_and_pre = condition.clone();
    let mut pre_post: Vec<PrePost> = vec![];
    let mut num_pre_post: Vec<AssignEffect> = vec![];

    for (&var, effects) in effects_by_variable {
        let orig_pre = condition.get(var).map_or(-1, |value| value as i32);
        let mut added_effect = false;

        for (&post, eff_conditions) in effects {
            let pre = orig_pre;
            // If the effect does not change the variable value, ignore it
            if pre == post as i32 {
                continue;
            }

            let mut eff_condition_lists: Vec<Vec<(usize, usize)>> = eff_conditions
                .iter()
                .map(Assignment::sorted_pairs)
                .collect();

            if ranges[var] == 2 {
                // Apply simplifications for binary variables
                if prune_stupid_effect_conditions(var, post, &mut eff_condition_lists) {
                    *simplified_effect_condition_counter += 1;
                }
            }

            for eff_condition in &eff_condition_lists {
                let mut filtered_eff_condition: Vec<(usize, usize)> = vec![];
                let mut eff_condition_contradicts = false;

                for &(variable, value) in eff_condition {
                    if let Some(prevail_val) = prevail_and_pre.get(variable) {
                        if prevail_val != value {
                            eff_condition_contradicts = true;
                            break;
                        }
                    } else {
                        filtered_eff_condition.push((variable, value));
                    }
                }

                if eff_condition_contradicts {
                    continue;
                }

                pre_post.push((var, pre, post, filtered_eff_condition));
                added_effect = true;
            }
        }

        if added_effect {
            prevail_and_pre.remove(var);
        }
    }

    for (&numvar, effects) in ass_effects_by_variable {
        for ((ass_op, post_var), eff_conditions) in effects {
            let eff_condition_lists: Vec<Vec<(usize, usize)>> = eff_conditions
                .iter()
                .map(Assignment::sorted_pairs)
                .collect();

            for eff_condition in &eff_condition_lists {
                num_pre_post.push((numvar, ass_op.clone(), *post_var, eff_condition.clone()));
            }
        }
    }

    if pre_post.is_empty() {
        // Check if any numeric effect is relevant
        let mut irrelevant = true;
        for (eff_var, _, _, _) in &num_pre_post {
            if relevant_numeric_variables.contains(eff_var) {
                irrelevant = false;
                break;
            }
        }
        if irrelevant {
            return None;
        }
    }

    // Remove effect variables from prevail
    let prevail = prevail_and_pre.into_pairs();

    Some(SASOperator::new(
        name.to_string(),
        prevail,
        pre_post,
        num_pre_post,
        cost,
    ))
}

// ============================================================
// prune_stupid_effect_conditions
// ============================================================

pub(super) fn prune_stupid_effect_conditions(
    var: usize,
    val: usize,
    conditions: &mut Vec<Vec<(usize, usize)>>,
) -> bool {
    if conditions == &[vec![]] {
        return false; // Quick exit for common case
    }

    assert!(val == 0 || val == 1);
    let dual_fact = (var, 1 - val);
    let mut simplified = false;

    for condition in conditions.iter_mut() {
        // Rule 1: remove dual fact from condition
        let len_before = condition.len();
        condition.retain(|f| *f != dual_fact);
        if condition.len() != len_before {
            simplified = true;
        }
        // Rule 2 is checked below
    }

    // Rule 2: if any condition is empty, simplify to [[]]
    if conditions.iter().any(|c| c.is_empty()) {
        *conditions = vec![vec![]];
        simplified = true;
    }

    simplified
}

// ============================================================
// translate_strips_axiom
// ============================================================

pub(super) fn translate_strips_axiom(
    translation: &mut Translation,
    axiom: &PropositionalAxiom,
) -> Vec<SASAxiom> {
    let conditions = translate_strips_conditions(translation, &axiom.condition);
    if conditions.is_none() {
        return vec![];
    }

    // Since issue454 every axiom *proves* its head: the rules that refute a
    // derived variable are computed by the consumer that needs them, from the SAS
    // task. A negated head reaching here would silently become a rule writing the
    // variable's `<none of those>` value, which is not the same fact.
    let Condition::Atom(atom) = &axiom.effect else {
        panic!("an axiom head is a positive atom, got {}", axiom.effect);
    };
    let Some(pairs) = translation.dictionary.get(atom) else {
        // The head has no reachable SAS fact. Every occurrence of that head in
        // a rule body was therefore simplified to false already, so omitting
        // this producer cannot remove a reachable grounded fact.
        return vec![];
    };
    let effect = pairs[0];

    let mut axioms = vec![];
    for condition in conditions.unwrap() {
        axioms.push(SASAxiom::new(condition.into_pairs(), effect));
    }
    axioms
}

// ============================================================
// translate_numeric_axiom
// ============================================================

pub(super) fn translate_numeric_axiom(
    axiom: &InstantiatedNumericAxiom,
    _prop_dictionary: &HashMap<Atom, Vec<(usize, usize)>>,
    num_dictionary: &HashMap<PrimitiveNumericExpression, usize>,
) -> Option<SASNumericAxiom> {
    // The one legitimate reason to drop an axiom: its effect has no SAS numeric
    // variable, because relevance analysis found nothing that reads it. Dropping
    // the definition of a variable that does not exist loses nothing. Every other
    // way out of this function is a panic, since a missing *operand*, a
    // surviving constant or a compound expression would each leave a variable
    // that does exist with no definition.
    let effect = num_dictionary.get(&axiom.effect)?;
    let op = &axiom.op;
    let mut parts = vec![];
    for part in &axiom.parts {
        match part {
            FunctionalExpression::PrimitiveNumericExpression(pne) => {
                let Some(&idx) = num_dictionary.get(pne) else {
                    panic!(
                        "numeric axiom for {} reads {pne}, which has no SAS numeric \
                         variable; dropping the axiom would leave its effect variable \
                         undefined",
                        axiom.effect
                    )
                };
                parts.push(idx);
            }
            // Both of these were `return None`, which dropped the axiom and left
            // the variable it defines with no definition at all. The comment on
            // the first already said the case was impossible, so saying so out
            // loud costs nothing and stops it being a silent wrong answer.
            FunctionalExpression::NumericConstant(constant) => panic!(
                "numeric axiom for {} still reads the constant {constant}; constants \
                 are folded before translation",
                axiom.effect
            ),
            other => panic!(
                "numeric axiom for {} reads the compound expression {other}; \
                 arithmetic is flattened into binary axioms before translation",
                axiom.effect
            ),
        }
    }
    Some(SASNumericAxiom::new(op.clone(), parts, *effect))
}

// ============================================================
// translate_strips_operators
// ============================================================

pub(super) fn translate_strips_operators(
    translation: &mut Translation,
    actions: &[PropositionalAction],
) -> Vec<SASOperator> {
    let mut result = vec![];
    for action in actions {
        let sas_ops = translate_strips_operator(translation, action);
        result.extend(sas_ops);
    }
    result
}

// ============================================================
// translate_strips_axioms
// ============================================================

pub(super) fn translate_strips_axioms(
    translation: &mut Translation,
    axioms: &[PropositionalAxiom],
) -> Vec<SASAxiom> {
    let mut result = vec![];
    for axiom in axioms {
        let sas_axioms = translate_strips_axiom(translation, axiom);
        result.extend(sas_axioms);
    }
    result
}

// ============================================================
// add_key_to_comp_axioms
// ============================================================

pub(super) fn add_key_to_comp_axioms(
    sas_comp_axioms: &[SASCompareAxiom],
    translation_key: &mut Vec<Vec<String>>,
) {
    for axiom in sas_comp_axioms {
        assert_eq!(
            axiom.effect,
            translation_key.len(),
            "current effect {} != next variable {}",
            axiom.effect,
            translation_key.len()
        );
        translation_key.push(vec![
            axiom.to_string(),
            axiom.invert_comparator().to_string(),
        ]);
    }
}
