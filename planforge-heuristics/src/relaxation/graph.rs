//! The relaxed planning graph FF explores: the fact universe -- the
//! propositional facts and the comparison axioms' TRUE values -- the relaxed
//! operators over it, including the synthetic operators of conditional
//! effects and the pseudo-operators of propositional axioms, and the achiever
//! and consumer indices the explorations walk.

use super::*;

/// The FF fact universe: one [`FactId`] per `(variable, value)` pair the
/// monotonic relaxation can represent.
///
/// Ordinary propositional variables contribute every value. A comparison
/// variable contributes only its `TRUE` value — the relaxation gains facts and
/// never loses them, so `FALSE` and `UNKNOWN` are unreachable once the axiom
/// has fired.
///
/// `AssignmentAxiom::get_affected_var_id` lives in the *numeric* index
/// namespace and must never reach the propositional bucket; conflating the two
/// namespaces silently dropped legitimate propositional facts in earlier
/// versions.
pub(super) struct FactUniverse {
    /// Flat `FactId` table, with one offset per propositional variable.
    pub(super) fact_id_by_var_value: Vec<Option<FactId>>,
    pub(super) fact_id_offsets: Vec<usize>,
    pub(super) var_value: Vec<(VariableIndex, ExplicitValueIndex)>,
    /// `to_axiom[fid]` is `Some(axiom_idx)` iff `fid` is a comparison axiom's
    /// TRUE fact.
    pub(super) to_axiom: Vec<Option<AxiomIdx>>,
    pub(super) comparison_axioms: Vec<ComparisonAxiomDesc>,
}

impl FactUniverse {
    pub(super) fn build(task: &Task) -> Result<Self, String> {
        let mut fact_id_offsets = Vec::with_capacity(task.variables().len());
        let mut flat_len = 0usize;
        for variable in task.variables() {
            fact_id_offsets.push(flat_len);
            flat_len = flat_len
                .checked_add(variable.domain_size())
                .ok_or_else(|| "FF fact table size overflow".to_string())?;
        }
        let mut fact_id_by_var_value = vec![None; flat_len];
        let mut var_value: Vec<(VariableIndex, ExplicitValueIndex)> = Vec::new();
        let mut to_axiom: Vec<Option<AxiomIdx>> = Vec::new();

        for (var_id, variable) in task.variables().iter().enumerate() {
            if task.variable_kind(VariableIndex::from_usize(var_id)) == VariableKind::Condition {
                continue;
            }
            let offset = fact_id_offsets[var_id];
            for value in 0..variable.domain_size() {
                fact_id_by_var_value[offset + value] = Some(var_value.len());
                var_value.push((
                    VariableIndex::from_usize(var_id),
                    ExplicitValueIndex::from_usize(value),
                ));
                to_axiom.push(None);
            }
        }

        let comparison_axioms = task.comparison_axioms();
        let mut descs = Vec::with_capacity(comparison_axioms.len());
        for (axiom_idx, axiom) in comparison_axioms.iter().enumerate() {
            let affected = axiom.get_affected_var_id();
            let &offset = fact_id_offsets.get(affected.index()).ok_or_else(|| {
                format!("comparison axiom {axiom_idx} affects out-of-range variable {affected}")
            })?;
            let true_value = ExplicitValueIndex::from_usize(ConditionValue::True.as_usize());
            let domain_size = task.variables()[affected.index()].domain_size();
            if true_value.index() >= domain_size {
                return Err(format!(
                    "comparison axiom {axiom_idx} affected variable has no TRUE value"
                ));
            }
            let slot = fact_id_by_var_value
                .get_mut(offset + true_value.index())
                .ok_or_else(|| {
                    format!("comparison axiom {axiom_idx} affected variable has no TRUE value")
                })?;
            let fid = var_value.len();
            *slot = Some(fid);
            var_value.push((affected, true_value));
            to_axiom.push(Some(AxiomIndex::from_usize(axiom_idx)));
            descs.push(ComparisonAxiomDesc {
                true_fact: fid,
                left_var: axiom.get_left_var_id(),
                right_var: axiom.get_right_var_id(),
                op: axiom.get_operator().clone(),
            });
        }

        Ok(Self {
            fact_id_by_var_value,
            fact_id_offsets,
            var_value,
            to_axiom,
            comparison_axioms: descs,
        })
    }

    pub(super) fn len(&self) -> usize {
        self.var_value.len()
    }

    #[inline]
    pub(super) fn fact_id(&self, fact: &ExplicitFact) -> Option<FactId> {
        self.fact_id_for_value(fact.var(), fact.value())
    }

    #[inline]
    pub(super) fn fact_id_for_value(&self, var: usize, value: usize) -> Option<FactId> {
        let &offset = self.fact_id_offsets.get(var)?;
        let end = self
            .fact_id_offsets
            .get(var + 1)
            .copied()
            .unwrap_or(self.fact_id_by_var_value.len());
        (value < end - offset)
            .then(|| self.fact_id_by_var_value[offset + value])
            .flatten()
    }

    /// Split `conditions` into the facts the relaxation represents and the
    /// [state-dependent](StateDependentPrecond) ones it cannot.
    pub(super) fn split_conditions(
        &self,
        conditions: &[ExplicitFact],
        facts: &mut Vec<FactId>,
        state_deps: &mut Vec<StateDependentPrecond>,
    ) {
        for condition in conditions {
            match self.fact_id(condition) {
                Some(fid) => facts.push(fid),
                None => state_deps.push((condition.var(), condition.value())),
            }
        }
    }
}

/// One entry of [`RelaxedOperators`], as the collection stage builds it.
pub(super) struct RelaxedOperator {
    /// Index into `task.operators()` for real operators; `None` for
    /// synthetic conditional-effect and propositional-axiom pseudo-ops.
    pub(super) task_idx: Option<usize>,
    /// Real operator whose cost this synthetic op charges; `None` for real
    /// operators and for zero-cost axiom pseudo-ops.
    pub(super) parent: Option<OpId>,
    pub(super) cost: f64,
    pub(super) preconditions: Vec<FactId>,
    pub(super) state_deps: Vec<StateDependentPrecond>,
    pub(super) effects: Vec<FactId>,
    pub(super) numeric_effects: Vec<AssignmentEffectDesc>,
}

/// Operators of the relaxed planning graph, as parallel arrays indexed by
/// [`OpId`]: one real op per task operator, one zero-cost synthetic op per
/// conditional effect, and one zero-cost pseudo-op per propositional axiom.
#[derive(Default)]
pub(super) struct RelaxedOperators {
    pub(super) task_idx: Vec<Option<usize>>,
    pub(super) parent: Vec<Option<OpId>>,
    pub(super) cost: Vec<f64>,
    pub(super) preconditions: Vec<Vec<FactId>>,
    pub(super) state_deps: Vec<Vec<StateDependentPrecond>>,
    pub(super) effects: Vec<Vec<FactId>>,
    pub(super) numeric_effects: Vec<Vec<AssignmentEffectDesc>>,
}

impl RelaxedOperators {
    pub(super) fn push(&mut self, op: RelaxedOperator) -> OpId {
        let op_id = self.preconditions.len();
        self.task_idx.push(op.task_idx);
        self.parent.push(op.parent);
        self.cost.push(op.cost);
        self.preconditions.push(op.preconditions);
        self.state_deps.push(op.state_deps);
        self.effects.push(op.effects);
        self.numeric_effects.push(op.numeric_effects);
        OperatorIndex::from_usize(op_id)
    }

    pub(super) fn len(&self) -> usize {
        self.preconditions.len()
    }
}

/// Initial value of every `Constant` numeric variable, `None` for the rest.
///
/// Captured at construction so effect directions can be classified statically.
pub(super) fn constant_numeric_values(task: &Task) -> Result<Vec<Option<NumericValue>>, String> {
    let initial_numeric = task.initial_numeric_values();
    task.numeric_variables()
        .iter()
        .enumerate()
        .map(|(idx, var)| match var.numeric_type() {
            NumericType::Constant => initial_numeric
                .get(idx)
                .copied()
                .map(Some)
                .ok_or_else(|| format!("constant numeric variable {idx} missing initial value")),
            _ => Ok(None),
        })
        .collect()
}

/// Which way an assignment effect can move the relaxation envelope.
///
/// Exact for a `Constant` right-hand side. For a non-constant one the signs
/// are not known statically, so the envelope is assumed bidirectional: that
/// widens the achiever set without losing soundness.
pub(super) fn direction_of_effect(
    op: &AssignmentOperation,
    rhs: NumVarId,
    constant_value: &[Option<NumericValue>],
) -> EffectDirection {
    let rhs_const = constant_value.get(rhs.index()).copied().flatten();
    match op {
        AssignmentOperation::Plus => match rhs_const {
            Some(v) if v > ZERO_VALUE => EffectDirection::GrowMax,
            Some(v) if v < ZERO_VALUE => EffectDirection::ShrinkMin,
            // Exact zero — no movement.
            Some(_) | None => EffectDirection::Both,
        },
        AssignmentOperation::Minus => match rhs_const {
            Some(v) if v > ZERO_VALUE => EffectDirection::ShrinkMin,
            Some(v) if v < ZERO_VALUE => EffectDirection::GrowMax,
            Some(_) | None => EffectDirection::Both,
        },
        AssignmentOperation::Assign => EffectDirection::Both,
        // Rejected by `assignment_effect_desc` before reaching here.
        AssignmentOperation::Times | AssignmentOperation::Divide => EffectDirection::Both,
    }
}

/// Describe one assignment effect, or `None` for the multiplicative
/// operations the monotonic relaxation cannot soundly bound. The caller
/// reports those with the context it has.
pub(super) fn assignment_effect_desc(
    assign: &AssignmentEffect,
    constant_value: &[Option<NumericValue>],
) -> Option<AssignmentEffectDesc> {
    match assign.operation() {
        AssignmentOperation::Plus | AssignmentOperation::Minus | AssignmentOperation::Assign => {
            Some(AssignmentEffectDesc {
                affected_var: assign.affected_var_id(),
                operation: assign.operation().clone(),
                rhs_var: assign.var_id(),
                direction: direction_of_effect(assign.operation(), assign.var_id(), constant_value),
            })
        }
        AssignmentOperation::Times | AssignmentOperation::Divide => None,
    }
}

/// Every task operator becomes one real op carrying its unconditional
/// effects, plus one zero-cost synthetic op per conditional effect. The
/// parent's state-dependent preconditions are inherited verbatim by each
/// synthetic; the conditional effect's own conditions are split the same way.
pub(super) fn add_task_operator(
    ops: &mut RelaxedOperators,
    universe: &FactUniverse,
    constant_value: &[Option<NumericValue>],
    task: &Task,
    op_idx: usize,
    op: &Operator,
) -> Result<(), String> {
    let mut parent_preconds: Vec<FactId> = Vec::new();
    let mut parent_state_deps: Vec<StateDependentPrecond> = Vec::new();
    universe.split_conditions(
        op.preconditions(),
        &mut parent_preconds,
        &mut parent_state_deps,
    );

    let mut parent_effects: Vec<FactId> = Vec::new();
    for eff in op.effects() {
        if !eff.conditions().is_empty() {
            continue;
        }
        if let Some(fid) = universe.fact_id(&ExplicitFact::propositional(
            eff.var_id().index(),
            eff.value().index(),
        )) {
            parent_effects.push(fid);
        }
    }

    let mut parent_numeric: Vec<AssignmentEffectDesc> = Vec::new();
    for assign in op.assignment_effects() {
        if !assign.conditions().is_empty() {
            continue;
        }
        let desc = assignment_effect_desc(assign, constant_value).ok_or_else(|| {
            format!(
                "operator {op_idx} (`{}`) uses unsupported {:?} assignment effect; \
                 the monotonic relaxation can't soundly bound it. Pick a different \
                 heuristic for tasks that need multiplicative numerics.",
                op.name(),
                assign.operation()
            )
        })?;
        parent_numeric.push(desc);
    }

    let parent_op_id = ops.push(RelaxedOperator {
        task_idx: Some(op_idx),
        parent: None,
        cost: task.operator_cost(OperatorIndex::from_usize(op_idx)),
        preconditions: parent_preconds.clone(),
        state_deps: parent_state_deps.clone(),
        effects: parent_effects,
        numeric_effects: parent_numeric,
    });

    for eff in op.effects() {
        if eff.conditions().is_empty() {
            continue;
        }
        let mut preconditions = parent_preconds.clone();
        let mut state_deps = parent_state_deps.clone();
        universe.split_conditions(eff.conditions(), &mut preconditions, &mut state_deps);
        let effects = universe
            .fact_id(&ExplicitFact::propositional(
                eff.var_id().index(),
                eff.value().index(),
            ))
            .into_iter()
            .collect();
        ops.push(RelaxedOperator {
            task_idx: None,
            parent: Some(parent_op_id),
            cost: 0.0,
            preconditions,
            state_deps,
            effects,
            numeric_effects: Vec::new(),
        });
    }

    for assign in op.assignment_effects() {
        if assign.conditions().is_empty() {
            continue;
        }
        let mut preconditions = parent_preconds.clone();
        let mut state_deps = parent_state_deps.clone();
        universe.split_conditions(assign.conditions(), &mut preconditions, &mut state_deps);
        let desc = assignment_effect_desc(assign, constant_value).ok_or_else(|| {
            format!(
                "operator {op_idx} (`{}`) uses unsupported {:?} conditional assignment effect.",
                op.name(),
                assign.operation()
            )
        })?;
        ops.push(RelaxedOperator {
            task_idx: None,
            parent: Some(parent_op_id),
            cost: 0.0,
            preconditions,
            state_deps,
            effects: Vec::new(),
            numeric_effects: vec![desc],
        });
    }

    Ok(())
}

/// A propositional axiom derives `(var_id, effect_value)` at no cost once its
/// conditions hold, so it is modelled as a zero-cost pseudo-operator.
///
/// Both values matter: the axiom fires for the transition
/// `precondition_value -> effect_value`, and the monotonic relaxation adds the
/// effect once. The precondition value joins the conditions, split the same
/// way — an `UNKNOWN` pre-value cannot be fabricated by the relaxation and so
/// becomes a state-dependent check.
pub(super) fn add_propositional_axiom_operator(
    ops: &mut RelaxedOperators,
    universe: &FactUniverse,
    axiom_idx: usize,
    axiom: &PropositionalAxiom,
) -> Result<(), String> {
    // An out-of-universe effect means the axiom drives a value of a
    // numeric-axiom variable, which the relaxation cannot represent.
    let effect_fid = universe
        .fact_id(&ExplicitFact::propositional(
            axiom.var_id().index(),
            axiom.effect_value().index(),
        ))
        .ok_or_else(|| {
            format!(
                "propositional axiom {axiom_idx} effect on \
                 variable {} value {} is unrepresentable in the FF \
                 fact universe (likely a numeric-axiom-driven variable)",
                axiom.var_id(),
                axiom.effect_value()
            )
        })?;

    let mut preconditions: Vec<FactId> = Vec::new();
    let mut state_deps: Vec<StateDependentPrecond> = Vec::new();
    universe.split_conditions(axiom.conditions(), &mut preconditions, &mut state_deps);
    universe.split_conditions(
        &[ExplicitFact::propositional(
            axiom.var_id().index(),
            axiom.precondition_value().index(),
        )],
        &mut preconditions,
        &mut state_deps,
    );

    ops.push(RelaxedOperator {
        task_idx: None,
        parent: None,
        cost: 0.0,
        preconditions,
        state_deps,
        effects: vec![effect_fid],
        numeric_effects: Vec::new(),
    });
    Ok(())
}

/// The task's operators, then its axioms, then the rules describing how a derived
/// variable takes its default value.
///
/// The last group is not in the task: the delete relaxation has no negation by
/// failure, so it has to be told how a derived variable becomes false, and the
/// rules that say so are exponential in the worst case and wanted by nothing but
/// a relaxation. See `planforge_sas::default_value_axioms`.
pub(super) fn collect_relaxed_operators(
    task: &Task,
    universe: &FactUniverse,
    constant_value: &[Option<NumericValue>],
) -> Result<RelaxedOperators, String> {
    let mut ops = RelaxedOperators::default();
    for (op_idx, op) in task.operators().iter().enumerate() {
        add_task_operator(&mut ops, universe, constant_value, task, op_idx, op)?;
    }
    let default_value_axioms =
        default_value_axioms(task, DefaultValueAxiomMode::ApproximateNegativeCycles);
    for (axiom_idx, axiom) in task
        .axioms()
        .iter()
        .chain(&default_value_axioms)
        .enumerate()
    {
        add_propositional_axiom_operator(&mut ops, universe, axiom_idx, axiom)?;
    }
    Ok(ops)
}

/// For each numeric variable, the comparison axioms naming it on either side.
///
/// Lets `fire_operator` re-evaluate only the axioms a numeric update can
/// affect.
pub(super) fn index_comparison_axioms_by_numeric_var(
    comparison_axioms: &[ComparisonAxiomDesc],
    num_numeric: usize,
) -> Result<Vec<Vec<AxiomIdx>>, String> {
    let mut by_var: Vec<Vec<AxiomIdx>> = vec![Vec::new(); num_numeric];
    for (idx, ax) in comparison_axioms.iter().enumerate() {
        if ax.left_var.index() >= num_numeric || ax.right_var.index() >= num_numeric {
            return Err(format!(
                "comparison axiom {idx} references out-of-range numeric variable \
                 (left={}, right={}, num_numeric={num_numeric})",
                ax.left_var, ax.right_var
            ));
        }
        by_var[ax.left_var.index()].push(AxiomIndex::from_usize(idx));
        if ax.right_var != ax.left_var {
            by_var[ax.right_var.index()].push(AxiomIndex::from_usize(idx));
        }
    }
    Ok(by_var)
}

/// Assignment axioms in topological (SAS axiom-layer) order, each describing a
/// derived numeric variable as `affected := left ∘ right`.
pub(super) fn collect_assignment_axioms(
    task: &Task,
    num_numeric: usize,
) -> Result<Vec<AssignmentAxiom>, String> {
    task.assignment_axioms()
        .iter()
        .enumerate()
        .map(|(axiom_idx, axiom)| {
            let affected = axiom.get_affected_var_id();
            if affected.index() >= num_numeric
                || axiom
                    .expression
                    .dependencies()
                    .any(|var| var.index() >= num_numeric)
            {
                return Err(format!(
                    "assignment axiom {axiom_idx} references an out-of-range numeric variable"
                ));
            }
            Ok(axiom.clone())
        })
        .collect()
}

/// For each numeric variable, the comparison axioms it reaches *through* one
/// or more assignment axioms but does not name directly.
///
/// This matters when a goal compares a derived numeric (`total_poured =
/// Σ poured`): real operators update the base variable and the assignment
/// axiom propagates into the derived one. The RPG forward pass already handles
/// that, but the achiever index used by `extract_relaxed_plan` matched only
/// direct effect-to-axiom-var hits, so derived-variable goals had empty
/// achiever lists, the relaxed plan came out empty, and FF degenerated to BFS
/// on tasks like plant-watering.
pub(super) fn comparison_axioms_via_derived_vars(
    comparison_axioms: &[ComparisonAxiomDesc],
    assignment_axioms: &[AssignmentAxiom],
    num_numeric: usize,
) -> Vec<Vec<AxiomIdx>> {
    let depends_on = compute_numeric_dependency_closure(num_numeric, assignment_axioms);
    let mut via_derived: Vec<Vec<AxiomIdx>> = vec![Vec::new(); num_numeric];
    for (idx, ax) in comparison_axioms.iter().enumerate() {
        for side in [ax.left_var, ax.right_var] {
            for &base in &depends_on[side.index()] {
                if base != ax.left_var
                    && base != ax.right_var
                    && !via_derived[base.index()].contains(&AxiomIndex::from_usize(idx))
                {
                    via_derived[base.index()].push(AxiomIndex::from_usize(idx));
                }
            }
        }
    }
    via_derived
}

/// For each fact, the operators that can achieve it under the monotonic
/// relaxation: for propositional facts the ops with it in their add list, and
/// for a comparison axiom's TRUE fact the ops whose numeric effects can push
/// the envelope the way the axiom needs.
///
/// The transitive registrations from [`comparison_axioms_via_derived_vars`]
/// skip the direction check: under the unbounded-firing relaxation Plus/Minus
/// effects push both sides to ±∞ anyway, and tracking sign flips through a
/// Difference chain would duplicate `propagate_assignment_axioms`. Registering
/// over-eagerly only adds candidates — the cheapest-supporter pick still has
/// to find them at a usable layer.
pub(super) fn build_achiever_index(
    operators: &RelaxedOperators,
    comparison_axioms: &[ComparisonAxiomDesc],
    axioms_touching_var: &[Vec<AxiomIdx>],
    axioms_via_derived: &[Vec<AxiomIdx>],
    num_facts: usize,
    num_numeric: usize,
) -> Result<Vec<Vec<OpId>>, String> {
    let mut achievers: Vec<Vec<OpId>> = vec![Vec::new(); num_facts];
    for (op_id, effs) in operators.effects.iter().enumerate() {
        for &fid in effs {
            achievers[fid].push(OperatorIndex::from_usize(op_id));
        }
    }

    let mut register = |true_fact: FactId, op_id: OpId| {
        if !achievers[true_fact].contains(&op_id) {
            achievers[true_fact].push(op_id);
        }
    };

    for (op_id, numeric_effs) in operators.numeric_effects.iter().enumerate() {
        for eff in numeric_effs {
            if eff.affected_var.index() >= num_numeric {
                return Err(format!(
                    "operator {op_id} effect on out-of-range numeric variable {}",
                    eff.affected_var
                ));
            }
            for &axiom_idx in &axioms_touching_var[eff.affected_var.index()] {
                let axiom = &comparison_axioms[axiom_idx.index()];
                if axiom_needs_direction(eff.affected_var, eff.direction, axiom) {
                    register(axiom.true_fact, OperatorIndex::from_usize(op_id));
                }
            }
            for &axiom_idx in &axioms_via_derived[eff.affected_var.index()] {
                register(
                    comparison_axioms[axiom_idx.index()].true_fact,
                    OperatorIndex::from_usize(op_id),
                );
            }
        }
    }
    Ok(achievers)
}

pub(super) fn collect_goal_facts(
    task: &Task,
    universe: &FactUniverse,
) -> Result<Vec<FactId>, String> {
    task.goals()
        .iter()
        .map(|goal| {
            universe.fact_id(goal).ok_or_else(|| {
                format!(
                    "goal fact {goal:?} maps to no FactId — variable {} value {} not \
                     in the FF fact universe (numeric-axiom non-TRUE goals are not \
                     representable under the delete relaxation)",
                    goal.var(),
                    goal.value()
                )
            })
        })
        .collect()
}

/// For each fact, the operators that have it as a precondition — the BFS's
/// forward edges.
pub(super) fn build_consumer_index(
    operators: &RelaxedOperators,
    num_facts: usize,
) -> Vec<Vec<OpId>> {
    let mut consumers: Vec<Vec<OpId>> = vec![Vec::new(); num_facts];
    for (op_id, prec) in operators.preconditions.iter().enumerate() {
        for &fid in prec {
            consumers[fid].push(OperatorIndex::from_usize(op_id));
        }
    }
    consumers
}
