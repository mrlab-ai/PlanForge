//! Fast-Forward (h_FF) heuristic with faithful Metric-FF numeric relaxation.
//!
//! Standard relaxed-plan heuristic of Hoffmann & Nebel (JAIR 2001), extended
//! per Hoffmann's Metric-FF (JAIR 2003) to handle numeric preconditions and
//! effects under a monotonic relaxation.
//!
//! # Algorithm
//!
//!  1. Build the relaxed planning graph layer by layer. Each numeric
//!     variable carries a `(max_reachable, min_reachable)` envelope updated
//!     by operator assignment effects. Comparison axioms (`(>= x v)` etc.)
//!     become available when the envelope makes them satisfiable.
//!  2. Stop once every goal fact is in the graph or no further progress is
//!     possible.
//!  3. Backward-chain greedy cheapest supporters from the goal layer; the
//!     extracted set of operators is the relaxed plan, and `h_FF(s)` is its
//!     summed cost.
//!
//! # Faithfulness vs. fast-path shortcuts
//!
//! This module does not silently drop or weaken constraints when input
//! semantics fall outside the modelled subset. Specifically:
//!
//!   * Conditional propositional and conditional numeric effects are
//!     expanded into "synthetic" pseudo-operators. A synthetic operator
//!     inherits the parent's propositional preconditions, adds the
//!     conditional effect's own conditions on top, and carries the
//!     conditional effect itself. Synthetic operators are zero-cost — they
//!     fire for free once their parent is in the plan — and adding one to
//!     the relaxed plan implicitly adds its parent for cost-counting
//!     purposes.
//!   * `Times` / `Divide` assignment effects are not soundly bounded by a
//!     direction-agnostic monotonic relaxation (sign-flips break it). The
//!     constructor returns an error if any such effect is encountered;
//!     callers must not request `ff()` on tasks that use these operations.
//!     Better an explicit error than a silently unsound heuristic.
//!   * Numeric-axiom-var preconditions on `FALSE` / `UNKNOWN` values are
//!     dropped from the relaxation — this is a *design property* of the
//!     delete relaxation (it only ever adds facts) rather than a fallback,
//!     and is the standard Metric-FF treatment.
//!
//! # Per-axiom achiever scoping
//!
//! For comparison-axiom TRUE facts, only operators whose numeric effects
//! actually move the envelope in the direction required by the axiom are
//! registered as candidate achievers. For `(>= x v)` the achievers are
//! operators that can grow `max[x]` or shrink `min[v]`; for `(<= x v)`
//! they are operators that can shrink `min[x]` or grow `max[v]`; for
//! `(== x v)` either direction qualifies; `(!= x v)` is trivially
//! satisfiable in the relaxation. The direction of an effect is computed
//! statically from the assignment operation and the right-hand-side
//! variable's type (`Constant` types give an exact sign; other types are
//! assumed bidirectional).
//!
//! References:
//!   * Hoffmann & Nebel, *The FF Planning System*, JAIR 2001.
//!   * Hoffmann, *The Metric-FF Planning System*, JAIR 2003.

use std::cell::RefCell;
use std::collections::{HashSet, VecDeque};

use crate::evaluation::evaluator::{EvaluationError, EvaluationState};
use crate::evaluation::heuristic::Heuristic;
use planforge_sas::axioms::{CalOperator, ComparisonOperator, PropositionalAxiom};
use planforge_sas::default_value_axioms::{DefaultValueAxiomMode, default_value_axioms};
use planforge_sas::numeric_conditions::ConditionValue;
use planforge_sas::numeric_task::{
    AbstractNumericTask, AssignmentEffect, AssignmentOperation, ExplicitFact, NumericType,
    Operator, metric_operator_cost_from_initial_values,
};
use planforge_sas::state_registry::StateRegistry;

type FactId = usize;
type OpId = usize;
type NumVarId = usize;
type AxiomIdx = usize;

/// Monotonic-relaxation envelope for one numeric variable.
#[derive(Debug, Clone, Copy)]
struct NumericRange {
    max: f64,
    min: f64,
}

impl NumericRange {
    const fn singleton(v: f64) -> Self {
        Self { max: v, min: v }
    }

    /// Returns `true` if `other` widens this range.
    fn join(&mut self, other: NumericRange) -> bool {
        let new_max = if other.max > self.max {
            other.max
        } else {
            self.max
        };
        let new_min = if other.min < self.min {
            other.min
        } else {
            self.min
        };
        // Use bit-pattern inequality rather than `> self.max + EPSILON` so
        // `+∞ vs finite max` reads as "widened" without an arithmetic-on-
        // infinity ambiguity.
        let changed =
            new_max.to_bits() != self.max.to_bits() || new_min.to_bits() != self.min.to_bits();
        self.max = new_max;
        self.min = new_min;
        changed
    }
}

/// Monotonic direction in which an assignment effect can push the affected
/// variable's envelope.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EffectDirection {
    /// Effect can grow `max[affected]` (and may also shrink `min[affected]`).
    GrowMax,
    /// Effect can shrink `min[affected]` (only).
    ShrinkMin,
    /// Effect could move either bound — typically because the RHS is a live
    /// numeric variable whose sign is not statically determined.
    Both,
}

impl EffectDirection {
    fn includes_grow_max(self) -> bool {
        matches!(self, EffectDirection::GrowMax | EffectDirection::Both)
    }
    fn includes_shrink_min(self) -> bool {
        matches!(self, EffectDirection::ShrinkMin | EffectDirection::Both)
    }
}

#[derive(Debug, Clone)]
struct AssignmentEffectDesc {
    affected_var: NumVarId,
    operation: AssignmentOperation,
    rhs_var: NumVarId,
    direction: EffectDirection,
}

/// Per-state-propagation descriptor for a single assignment axiom. The
/// axiom computes `affected := left ∘ right` where `∘` is `Sum` or
/// `Difference`. Multiplicative axioms (`Product` / `Division`) are
/// rejected at construction — they don't admit a sign-agnostic monotonic
/// bound.
#[derive(Debug, Clone)]
struct AssignmentAxiomDesc {
    affected_var: NumVarId,
    left_var: NumVarId,
    right_var: NumVarId,
    op: CalOperator,
}

#[derive(Debug, Clone)]
struct ComparisonAxiomDesc {
    /// FactId of the TRUE-value fact for the propositional variable backing
    /// this axiom.
    true_fact: FactId,
    left_var: NumVarId,
    right_var: NumVarId,
    op: ComparisonOperator,
}

/// Per-state RPG buffers reused across `compute_heuristic` calls.
struct ScratchBuffers {
    fact_first_layer: Vec<i32>,
    op_remaining_preconditions: Vec<i32>,
    op_first_layer: Vec<i32>,
    queue: VecDeque<FactId>,
    goals_at_layer: Vec<Vec<FactId>>,
    seen: Vec<bool>,
    in_plan: Vec<bool>,
    /// Per-evaluation operator eligibility — `false` for ops whose
    /// state-dependent preconditions don't hold in the current state.
    /// Ineligible ops are skipped throughout the BFS and ignored as
    /// achievers during relaxed-plan extraction.
    op_eligible: Vec<bool>,
    numeric: Vec<NumericRange>,
    axiom_first_layer: Vec<i32>,
    /// Reusable Vec for "numeric vars dirtied during the current
    /// operator firing". Cleared at the start of each `fire_operator`
    /// call so it doesn't accumulate.
    dirty_vars_scratch: Vec<NumVarId>,
    /// Reusable Vec for "comparison axioms to re-evaluate during the
    /// current operator firing". Cleared at the start of each call.
    dirty_axioms_scratch: Vec<AxiomIdx>,
    /// Bitset over numeric vars, used by `fire_operator` to dedup the
    /// push into `dirty_vars_scratch`. Entries are cleared in pairs with
    /// the corresponding push so there's no global reset cost.
    dirty_var_mark: Vec<bool>,
    /// Bitset over comparison axioms, paired with `dirty_axioms_scratch`.
    dirty_axiom_mark: Vec<bool>,
    /// Reused propositional-state buffer for layer-0 fact resolution.
    /// Eliminates the per-fact `state_packer.get` call by reading the
    /// entire packed state once and indexing into the resulting Vec.
    prop_state_values: Vec<usize>,
}

impl ScratchBuffers {
    fn new(num_facts: usize, num_ops: usize, num_numeric: usize, num_axioms: usize) -> Self {
        Self {
            fact_first_layer: vec![-1; num_facts],
            op_remaining_preconditions: Vec::with_capacity(num_ops),
            op_first_layer: vec![-1; num_ops],
            queue: VecDeque::new(),
            goals_at_layer: Vec::new(),
            seen: vec![false; num_facts],
            in_plan: vec![false; num_ops],
            op_eligible: vec![true; num_ops],
            numeric: vec![NumericRange::singleton(0.0); num_numeric],
            axiom_first_layer: vec![-1; num_axioms],
            dirty_vars_scratch: Vec::new(),
            dirty_axioms_scratch: Vec::new(),
            dirty_var_mark: vec![false; num_numeric],
            dirty_axiom_mark: vec![false; num_axioms],
            prop_state_values: Vec::new(),
        }
    }

    fn reset(&mut self) {
        for v in &mut self.fact_first_layer {
            *v = -1;
        }
        self.op_remaining_preconditions.clear();
        for v in &mut self.op_first_layer {
            *v = -1;
        }
        self.queue.clear();
        self.goals_at_layer.clear();
        for v in &mut self.seen {
            *v = false;
        }
        for v in &mut self.in_plan {
            *v = false;
        }
        for v in &mut self.op_eligible {
            *v = true;
        }
        for v in &mut self.axiom_first_layer {
            *v = -1;
        }
    }
}

/// A propositional precondition of an operator whose `(var, value)` doesn't
/// have a `FactId` in the FF universe — typically a comparison-axiom
/// variable at its `FALSE` or `UNKNOWN` value. Under monotonic relaxation
/// these can only be satisfied at layer 0: once the axiom's TRUE fact is
/// derived (or any other fact added) the relaxation cannot un-derive it.
/// We therefore check them against the live initial state at evaluation
/// time and disable the operator outright if any fails to hold.
type StateDependentPrecond = (usize, usize);

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
struct FactUniverse {
    /// `id_by_var_value[var][value]`, `None` outside the universe.
    id_by_var_value: Vec<Vec<Option<FactId>>>,
    var_value: Vec<(usize, usize)>,
    /// `to_axiom[fid]` is `Some(axiom_idx)` iff `fid` is a comparison axiom's
    /// TRUE fact.
    to_axiom: Vec<Option<AxiomIdx>>,
    comparison_axioms: Vec<ComparisonAxiomDesc>,
}

impl FactUniverse {
    fn build(task: &dyn AbstractNumericTask) -> Result<Self, String> {
        let mut id_by_var_value: Vec<Vec<Option<FactId>>> = task
            .variables()
            .iter()
            .map(|variable| vec![None; variable.domain_size()])
            .collect();
        let mut var_value: Vec<(usize, usize)> = Vec::new();
        let mut to_axiom: Vec<Option<AxiomIdx>> = Vec::new();

        for (var_id, values) in id_by_var_value.iter_mut().enumerate() {
            if task.numeric_conditions().is_condition_var(var_id) {
                continue;
            }
            for (value, fact_id) in values.iter_mut().enumerate() {
                *fact_id = Some(var_value.len());
                var_value.push((var_id, value));
                to_axiom.push(None);
            }
        }

        let comparison_axioms = task.comparison_axioms();
        let mut descs = Vec::with_capacity(comparison_axioms.len());
        for (axiom_idx, axiom) in comparison_axioms.iter().enumerate() {
            let affected = axiom.get_affected_var_id();
            let row = id_by_var_value.get_mut(affected).ok_or_else(|| {
                format!("comparison axiom {axiom_idx} affects out-of-range variable {affected}")
            })?;
            let true_value = ConditionValue::True.as_usize();
            let slot = row.get_mut(true_value).ok_or_else(|| {
                format!("comparison axiom {axiom_idx} affected variable has no TRUE value")
            })?;
            let fid = var_value.len();
            *slot = Some(fid);
            var_value.push((affected, true_value));
            to_axiom.push(Some(axiom_idx));
            descs.push(ComparisonAxiomDesc {
                true_fact: fid,
                left_var: axiom.get_left_var_id(),
                right_var: axiom.get_right_var_id(),
                op: axiom.get_operator().clone(),
            });
        }

        Ok(Self {
            id_by_var_value,
            var_value,
            to_axiom,
            comparison_axioms: descs,
        })
    }

    fn len(&self) -> usize {
        self.var_value.len()
    }

    #[inline]
    fn fact_id(&self, fact: &ExplicitFact) -> Option<FactId> {
        *self.id_by_var_value.get(fact.var())?.get(fact.value())?
    }

    /// Split `conditions` into the facts the relaxation represents and the
    /// [state-dependent](StateDependentPrecond) ones it cannot.
    fn split_conditions(
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
struct RelaxedOperator {
    /// Index into `task.get_operators()` for real operators; `None` for
    /// synthetic conditional-effect and propositional-axiom pseudo-ops.
    task_idx: Option<usize>,
    /// Real operator whose cost this synthetic op charges; `None` for real
    /// operators and for zero-cost axiom pseudo-ops.
    parent: Option<OpId>,
    cost: f64,
    preconditions: Vec<FactId>,
    state_deps: Vec<StateDependentPrecond>,
    effects: Vec<FactId>,
    numeric_effects: Vec<AssignmentEffectDesc>,
}

/// Operators of the relaxed planning graph, as parallel arrays indexed by
/// [`OpId`]: one real op per task operator, one zero-cost synthetic op per
/// conditional effect, and one zero-cost pseudo-op per propositional axiom.
#[derive(Default)]
struct RelaxedOperators {
    task_idx: Vec<Option<usize>>,
    parent: Vec<Option<OpId>>,
    cost: Vec<f64>,
    preconditions: Vec<Vec<FactId>>,
    state_deps: Vec<Vec<StateDependentPrecond>>,
    effects: Vec<Vec<FactId>>,
    numeric_effects: Vec<Vec<AssignmentEffectDesc>>,
}

impl RelaxedOperators {
    fn push(&mut self, op: RelaxedOperator) -> OpId {
        let op_id = self.preconditions.len();
        self.task_idx.push(op.task_idx);
        self.parent.push(op.parent);
        self.cost.push(op.cost);
        self.preconditions.push(op.preconditions);
        self.state_deps.push(op.state_deps);
        self.effects.push(op.effects);
        self.numeric_effects.push(op.numeric_effects);
        op_id
    }

    fn len(&self) -> usize {
        self.preconditions.len()
    }
}

/// Initial value of every `Constant` numeric variable, `None` for the rest.
///
/// Captured at construction so effect directions can be classified statically.
fn constant_numeric_values(task: &dyn AbstractNumericTask) -> Result<Vec<Option<f64>>, String> {
    let initial_numeric = task.get_initial_numeric_state_values();
    task.numeric_variables()
        .iter()
        .enumerate()
        .map(|(idx, var)| match var.get_type() {
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
fn direction_of_effect(
    op: &AssignmentOperation,
    rhs: NumVarId,
    constant_value: &[Option<f64>],
) -> EffectDirection {
    let rhs_const = constant_value.get(rhs).copied().flatten();
    match op {
        AssignmentOperation::Plus => match rhs_const {
            Some(v) if v > 0.0 => EffectDirection::GrowMax,
            Some(v) if v < 0.0 => EffectDirection::ShrinkMin,
            // Exact zero — no movement.
            Some(_) | None => EffectDirection::Both,
        },
        AssignmentOperation::Minus => match rhs_const {
            Some(v) if v > 0.0 => EffectDirection::ShrinkMin,
            Some(v) if v < 0.0 => EffectDirection::GrowMax,
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
fn assignment_effect_desc(
    assign: &AssignmentEffect,
    constant_value: &[Option<f64>],
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
fn add_task_operator(
    ops: &mut RelaxedOperators,
    universe: &FactUniverse,
    constant_value: &[Option<f64>],
    task: &dyn AbstractNumericTask,
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
        if let Some(fid) = universe.fact_id(&ExplicitFact::propositional(eff.var_id(), eff.value()))
        {
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
        cost: metric_operator_cost_from_initial_values(task, op).max(0.0),
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
            .fact_id(&ExplicitFact::propositional(eff.var_id(), eff.value()))
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
fn add_propositional_axiom_operator(
    ops: &mut RelaxedOperators,
    universe: &FactUniverse,
    axiom_idx: usize,
    axiom: &PropositionalAxiom,
) -> Result<(), String> {
    // An out-of-universe effect means the axiom drives a value of a
    // numeric-axiom variable, which the relaxation cannot represent.
    let effect_fid = universe
        .fact_id(&ExplicitFact::propositional(
            axiom.var_id(),
            axiom.effect_value(),
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
            axiom.var_id(),
            axiom.precondition_value(),
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
fn collect_relaxed_operators(
    task: &dyn AbstractNumericTask,
    universe: &FactUniverse,
    constant_value: &[Option<f64>],
) -> Result<RelaxedOperators, String> {
    let mut ops = RelaxedOperators::default();
    for (op_idx, op) in task.get_operators().iter().enumerate() {
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
fn index_comparison_axioms_by_numeric_var(
    comparison_axioms: &[ComparisonAxiomDesc],
    num_numeric: usize,
) -> Result<Vec<Vec<AxiomIdx>>, String> {
    let mut by_var: Vec<Vec<AxiomIdx>> = vec![Vec::new(); num_numeric];
    for (idx, ax) in comparison_axioms.iter().enumerate() {
        if ax.left_var >= num_numeric || ax.right_var >= num_numeric {
            return Err(format!(
                "comparison axiom {idx} references out-of-range numeric variable \
                 (left={}, right={}, num_numeric={num_numeric})",
                ax.left_var, ax.right_var
            ));
        }
        by_var[ax.left_var].push(idx);
        if ax.right_var != ax.left_var {
            by_var[ax.right_var].push(idx);
        }
    }
    Ok(by_var)
}

/// Assignment axioms in topological (SAS axiom-layer) order, each describing a
/// derived numeric variable as `affected := left ∘ right`.
fn collect_assignment_axioms(
    task: &dyn AbstractNumericTask,
    num_numeric: usize,
) -> Result<Vec<AssignmentAxiomDesc>, String> {
    task.assignment_axioms()
        .iter()
        .enumerate()
        .map(|(axiom_idx, axiom)| {
            let affected = axiom.get_affected_var_id();
            let left = axiom.get_left_var_id();
            let right = axiom.get_right_var_id();
            if affected >= num_numeric || left >= num_numeric || right >= num_numeric {
                return Err(format!(
                    "assignment axiom {axiom_idx} references out-of-range numeric variable \
                     (affected={affected}, left={left}, right={right}, num_numeric={num_numeric})"
                ));
            }
            Ok(AssignmentAxiomDesc {
                affected_var: affected,
                left_var: left,
                right_var: right,
                op: axiom.get_operator().clone(),
            })
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
fn comparison_axioms_via_derived_vars(
    comparison_axioms: &[ComparisonAxiomDesc],
    assignment_axioms: &[AssignmentAxiomDesc],
    num_numeric: usize,
) -> Vec<Vec<AxiomIdx>> {
    let depends_on = compute_numeric_dependency_closure(num_numeric, assignment_axioms);
    let mut via_derived: Vec<Vec<AxiomIdx>> = vec![Vec::new(); num_numeric];
    for (idx, ax) in comparison_axioms.iter().enumerate() {
        for side in [ax.left_var, ax.right_var] {
            for &base in &depends_on[side] {
                if base != ax.left_var && base != ax.right_var && !via_derived[base].contains(&idx)
                {
                    via_derived[base].push(idx);
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
fn build_achiever_index(
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
            achievers[fid].push(op_id);
        }
    }

    let mut register = |true_fact: FactId, op_id: OpId| {
        if !achievers[true_fact].contains(&op_id) {
            achievers[true_fact].push(op_id);
        }
    };

    for (op_id, numeric_effs) in operators.numeric_effects.iter().enumerate() {
        for eff in numeric_effs {
            if eff.affected_var >= num_numeric {
                return Err(format!(
                    "operator {op_id} effect on out-of-range numeric variable {}",
                    eff.affected_var
                ));
            }
            for &axiom_idx in &axioms_touching_var[eff.affected_var] {
                let axiom = &comparison_axioms[axiom_idx];
                if axiom_needs_direction(eff.affected_var, eff.direction, axiom) {
                    register(axiom.true_fact, op_id);
                }
            }
            for &axiom_idx in &axioms_via_derived[eff.affected_var] {
                register(comparison_axioms[axiom_idx].true_fact, op_id);
            }
        }
    }
    Ok(achievers)
}

fn collect_goal_facts(
    task: &dyn AbstractNumericTask,
    universe: &FactUniverse,
) -> Result<Vec<FactId>, String> {
    (0..task.get_num_goals())
        .map(|i| {
            let goal = task.get_goal_fact(i);
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
fn build_consumer_index(operators: &RelaxedOperators, num_facts: usize) -> Vec<Vec<OpId>> {
    let mut consumers: Vec<Vec<OpId>> = vec![Vec::new(); num_facts];
    for (op_id, prec) in operators.preconditions.iter().enumerate() {
        for &fid in prec {
            consumers[fid].push(op_id);
        }
    }
    consumers
}

pub struct FfHeuristic<'task> {
    /// Live borrow of the task — used to return cloned `Operator`s for the
    /// helpful-action interface.
    task: &'task dyn AbstractNumericTask,
    /// For each (real or synthetic) operator in `op_preconditions`, the
    /// index into `task.get_operators()` if it's a real operator (used for
    /// helpful-action reporting), `None` for synthetic conditional-effect
    /// pseudo-ops and propositional-axiom pseudo-ops (neither corresponds
    /// to a task operator the search engine can execute directly).
    op_task_idx: Vec<Option<usize>>,
    /// Per-(real or synthetic)-operator propositional preconditions.
    op_preconditions: Vec<Vec<FactId>>,
    /// Per-operator preconditions whose value is not representable in the
    /// FF universe (e.g. comparison-axiom FALSE). Checked at evaluation
    /// time against the live state; if any fails the operator is excluded
    /// from the RPG for that state. Not silently dropped.
    op_state_deps: Vec<Vec<StateDependentPrecond>>,
    /// Per-operator propositional add-effects.
    op_effects: Vec<Vec<FactId>>,
    /// Per-operator monotonic numeric effects.
    op_numeric_effects: Vec<Vec<AssignmentEffectDesc>>,
    /// Real cost of each operator. Synthetic (conditional-effect) ops are
    /// `0` — their parent's cost is paid via `op_parent`.
    op_cost: Vec<f64>,
    /// For each synthetic op, the real-op index whose cost should be paid
    /// when this synthetic appears in the relaxed plan. `None` for real
    /// ops; `Some(parent_real_op_id)` for synthetics.
    op_parent: Vec<Option<OpId>>,
    goal_facts: Vec<FactId>,
    /// For each fact, list of operators that can achieve it under the
    /// monotonic relaxation. For propositional facts: ops with that fact
    /// in their add-list. For comparison-axiom TRUE facts: ops whose
    /// numeric effects can push the envelope in the direction the axiom
    /// requires (see `register_axiom_achievers`).
    achievers: Vec<Vec<OpId>>,
    /// For each fact id, the operators that have it as a precondition.
    consumers: Vec<Vec<OpId>>,
    fact_var_value: Vec<(usize, usize)>,
    /// `fact_to_axiom[fid]` is `Some(axiom_idx)` iff this fact represents
    /// a comparison-axiom TRUE value; `None` for ordinary prop facts.
    fact_to_axiom: Vec<Option<AxiomIdx>>,
    comparison_axioms: Vec<ComparisonAxiomDesc>,
    /// Assignment axioms in topological (SAS axiom-layer) order. Each
    /// describes a derived numeric variable as `affected := left ∘ right`
    /// for `∘ ∈ {Sum, Difference}`.
    assignment_axioms: Vec<AssignmentAxiomDesc>,
    /// For each numeric var, indices of comparison axioms whose LHS or
    /// RHS mentions it. Lets `fire_operator` re-evaluate only the affected
    /// comparison axioms after a numeric update.
    axioms_touching_var: Vec<Vec<AxiomIdx>>,
    num_facts: usize,
    num_numeric: usize,
    scratch: RefCell<ScratchBuffers>,
    /// Cache of the task-operator indices of helpful actions for the most
    /// recently evaluated state. Populated at the end of every
    /// `compute_heuristic`. The search engine reads it once per state via
    /// `get_preferred_operator_ids` and stores the IDs on the search node;
    /// from there on it does integer-membership tests, not `Operator`
    /// clones. `get_preferred_operators` (the `Operator`-returning trait
    /// method) is still implemented by cloning from the task on demand,
    /// for callers that want full operator objects.
    last_helpful_action_ids: RefCell<Vec<usize>>,
}

impl<'task> FfHeuristic<'task> {
    /// Compile the task into the relaxed planning graph this heuristic
    /// evaluates: a fact universe, the operators over it, and the achiever and
    /// consumer indices the BFS and relaxed-plan extraction walk.
    pub fn new(task: &'task dyn AbstractNumericTask) -> Result<Self, String> {
        let universe = FactUniverse::build(task)?;
        let num_facts = universe.len();
        let num_numeric = task.numeric_variables().len();

        let axioms_touching_var =
            index_comparison_axioms_by_numeric_var(&universe.comparison_axioms, num_numeric)?;
        let assignment_axioms = collect_assignment_axioms(task, num_numeric)?;
        let constant_value = constant_numeric_values(task)?;

        let operators = collect_relaxed_operators(task, &universe, &constant_value)?;
        let num_ops = operators.len();

        let axioms_via_derived = comparison_axioms_via_derived_vars(
            &universe.comparison_axioms,
            &assignment_axioms,
            num_numeric,
        );
        let achievers = build_achiever_index(
            &operators,
            &universe.comparison_axioms,
            &axioms_touching_var,
            &axioms_via_derived,
            num_facts,
            num_numeric,
        )?;
        let goal_facts = collect_goal_facts(task, &universe)?;
        let consumers = build_consumer_index(&operators, num_facts);

        let num_comparison_axioms = universe.comparison_axioms.len();
        Ok(Self {
            task,
            op_task_idx: operators.task_idx,
            op_preconditions: operators.preconditions,
            op_state_deps: operators.state_deps,
            op_effects: operators.effects,
            op_numeric_effects: operators.numeric_effects,
            op_cost: operators.cost,
            op_parent: operators.parent,
            goal_facts,
            achievers,
            consumers,
            fact_var_value: universe.var_value,
            fact_to_axiom: universe.to_axiom,
            comparison_axioms: universe.comparison_axioms,
            assignment_axioms,
            axioms_touching_var,
            num_facts,
            num_numeric,
            scratch: RefCell::new(ScratchBuffers::new(
                num_facts,
                num_ops,
                num_numeric,
                num_comparison_axioms,
            )),
            last_helpful_action_ids: RefCell::new(Vec::new()),
        })
    }

    fn initial_numeric_state(
        &self,
        eval_state: &EvaluationState<'_, '_>,
        registry: &StateRegistry<'_>,
    ) -> Result<Vec<NumericRange>, EvaluationError> {
        let mut buffer: Vec<f64> = Vec::new();
        registry
            .fill_numeric_vars(eval_state.state(), &mut buffer)
            .map_err(|err| {
                EvaluationError::ComputationFailed(format!(
                    "FF heuristic failed to read numeric state: {err:?}"
                ))
            })?;
        if buffer.len() != self.num_numeric {
            return Err(EvaluationError::ComputationFailed(format!(
                "FF heuristic: numeric-state length ({}) disagrees with task numeric-variable \
                 count ({})",
                buffer.len(),
                self.num_numeric
            )));
        }
        Ok(buffer.into_iter().map(NumericRange::singleton).collect())
    }

    /// Propagate updated bounds through all assignment axioms until fixed
    /// point. Each pass refreshes derived numeric ranges; subsequent passes
    /// catch dependencies among derived vars. Pushes any numeric var
    /// whose range was widened into `dirty_out` (dedup against
    /// `dirty_mark`, which the caller is responsible for sizing and
    /// clearing). Returns whether *anything* widened.
    fn propagate_assignment_axioms(
        &self,
        numeric: &mut [NumericRange],
        dirty_out: &mut Vec<NumVarId>,
        dirty_mark: &mut [bool],
    ) -> bool {
        let mut any_change = false;
        loop {
            let mut changed = false;
            for ax in &self.assignment_axioms {
                let l = numeric[ax.left_var];
                let r = numeric[ax.right_var];
                let new = match ax.op {
                    CalOperator::Sum => NumericRange {
                        max: l.max + r.max,
                        min: l.min + r.min,
                    },
                    CalOperator::Difference => NumericRange {
                        max: l.max - r.min,
                        min: l.min - r.max,
                    },
                    // Sign-aware interval multiplication for `d := l * r`.
                    // With l ∈ [l.min, l.max] and r ∈ [r.min, r.max], the
                    // resulting range is bracketed by the four corner
                    // products. Each product is `extreme * extreme`, so
                    // any value of l*r lies in the [min, max] of those
                    // four. The monotonic-relaxation join is a union with
                    // the existing envelope, so widening is admissible.
                    CalOperator::Product => {
                        let p1 = l.min * r.min;
                        let p2 = l.min * r.max;
                        let p3 = l.max * r.min;
                        let p4 = l.max * r.max;
                        NumericRange {
                            max: p1.max(p2).max(p3).max(p4),
                            min: p1.min(p2).min(p3).min(p4),
                        }
                    }
                    // Sign-aware interval division for `d := l / r`. If
                    // r's interval contains zero, the quotient is
                    // unbounded — yield `[-∞, +∞]` (still admissible).
                    // Otherwise the four corner quotients bracket the
                    // result the same way Product does.
                    CalOperator::Division => {
                        if r.min <= 0.0 && 0.0 <= r.max {
                            NumericRange {
                                max: f64::INFINITY,
                                min: f64::NEG_INFINITY,
                            }
                        } else {
                            let q1 = l.min / r.min;
                            let q2 = l.min / r.max;
                            let q3 = l.max / r.min;
                            let q4 = l.max / r.max;
                            NumericRange {
                                max: q1.max(q2).max(q3).max(q4),
                                min: q1.min(q2).min(q3).min(q4),
                            }
                        }
                    }
                };
                if numeric[ax.affected_var].join(new) {
                    if !dirty_mark[ax.affected_var] {
                        dirty_mark[ax.affected_var] = true;
                        dirty_out.push(ax.affected_var);
                    }
                    changed = true;
                    any_change = true;
                }
            }
            if !changed {
                break;
            }
        }
        any_change
    }

    fn evaluate_axiom(&self, axiom: &ComparisonAxiomDesc, numeric: &[NumericRange]) -> bool {
        // `axiom.left_var` / `right_var` were range-checked at construction
        // (see step 3); a panic here would mean a corrupt heuristic.
        let l = numeric[axiom.left_var];
        let r = numeric[axiom.right_var];
        match axiom.op {
            ComparisonOperator::LessThan => l.min < r.max,
            ComparisonOperator::LessThanOrEqual => l.min <= r.max,
            ComparisonOperator::Equal => l.min <= r.max && l.max >= r.min,
            ComparisonOperator::GreaterThanOrEqual => l.max >= r.min,
            ComparisonOperator::GreaterThan => l.max > r.min,
            ComparisonOperator::UnEqual => l.min != l.max || r.min != r.max || l.min != r.min,
        }
    }

    fn apply_numeric_effect(
        &self,
        eff: &AssignmentEffectDesc,
        numeric: &mut [NumericRange],
    ) -> bool {
        // Indices range-checked at construction (steps 3 & 6).
        //
        // Standard Metric-FF semantics: each numeric-grow operator fires
        // *unboundedly many times* in the delete relaxation. So `Plus(var,
        // +k)` reachability-wise pushes `max[var]` to `+∞`, not just by
        // `+k`. Without this, "need N pours" can't be relaxed after one
        // firing — the RPG stalls below the threshold and the heuristic
        // declares dead-ends.
        //
        // `Assign(var, rhs)` is *not* iterable in the same sense — it
        // overwrites once — so it stays at the range-union semantics.
        let rhs = numeric[eff.rhs_var];
        let prev = numeric[eff.affected_var];
        let new = match eff.operation {
            AssignmentOperation::Assign => NumericRange {
                max: prev.max.max(rhs.max),
                min: prev.min.min(rhs.min),
            },
            AssignmentOperation::Plus => {
                let mut next = prev;
                if rhs.max > 0.0 {
                    next.max = f64::INFINITY;
                }
                if rhs.min < 0.0 {
                    next.min = f64::NEG_INFINITY;
                }
                next
            }
            AssignmentOperation::Minus => {
                let mut next = prev;
                if rhs.min < 0.0 {
                    next.max = f64::INFINITY;
                }
                if rhs.max > 0.0 {
                    next.min = f64::NEG_INFINITY;
                }
                next
            }
            AssignmentOperation::Times | AssignmentOperation::Divide => {
                unreachable!(
                    "Times/Divide assignment effects should have been rejected at construction"
                );
            }
        };
        numeric[eff.affected_var].join(new)
    }

    fn build_rpg(
        &self,
        eval_state: &EvaluationState<'_, '_>,
        registry: &StateRegistry<'_>,
        scratch: &mut ScratchBuffers,
    ) -> Result<i32, EvaluationError> {
        // Operator eligibility from state-dependent preconditions. An op
        // whose `(var, value)` precond is unrepresentable in the FF
        // universe (typically a comparison-axiom FALSE / UNKNOWN value)
        // is admissible in the relaxation iff the precondition is
        // satisfied in the live state — the monotonic relaxation cannot
        // make it true later. Mark such ops ineligible up front.
        scratch
            .op_eligible
            .resize(self.op_preconditions.len(), true);
        let live_state = eval_state.state();
        for (op_id, deps) in self.op_state_deps.iter().enumerate() {
            if deps.is_empty() {
                continue;
            }
            let eligible = deps.iter().all(|&(var, value)| {
                ExplicitFact::propositional(var, value).is_hold(live_state, registry)
            });
            scratch.op_eligible[op_id] = eligible;
        }

        scratch.numeric = self.initial_numeric_state(eval_state, registry)?;
        // The initial state already evaluates derived numerics correctly,
        // but `fill_numeric_vars` returns singleton ranges for them. Run
        // assignment-axiom propagation once so any wider-than-singleton
        // bounds (e.g. uninitialized derived = -∞/+∞) settle to a
        // consistent starting point. We don't care about which vars
        // widened at this point — discard the dirty list afterwards.
        scratch.dirty_vars_scratch.clear();
        let _ = self.propagate_assignment_axioms(
            &mut scratch.numeric,
            &mut scratch.dirty_vars_scratch,
            &mut scratch.dirty_var_mark,
        );
        for &v in &scratch.dirty_vars_scratch {
            scratch.dirty_var_mark[v] = false;
        }
        scratch.dirty_vars_scratch.clear();

        // Layer 0 propositional facts. Batch-read the entire packed
        // propositional state once (`fill_state`), then index directly
        // into the resulting Vec — saves O(num_facts) bound-checked
        // `state_packer.get` calls per evaluation.
        live_state.fill_state(registry, &mut scratch.prop_state_values);
        for fid in 0..self.num_facts {
            if self.fact_to_axiom[fid].is_some() {
                continue;
            }
            let (var, value) = self.fact_var_value[fid];
            if scratch
                .prop_state_values
                .get(var)
                .is_some_and(|v| *v == value)
            {
                scratch.fact_first_layer[fid] = 0;
                scratch.queue.push_back(fid);
            }
        }

        // Layer 0 comparison-axiom TRUE facts.
        for (axiom_idx, axiom) in self.comparison_axioms.iter().enumerate() {
            if self.evaluate_axiom(axiom, &scratch.numeric) {
                if scratch.fact_first_layer[axiom.true_fact] < 0 {
                    scratch.fact_first_layer[axiom.true_fact] = 0;
                    scratch.queue.push_back(axiom.true_fact);
                }
                scratch.axiom_first_layer[axiom_idx] = 0;
            }
        }

        // Reset per-op remaining-precondition counters.
        scratch
            .op_remaining_preconditions
            .resize(self.op_preconditions.len(), 0);
        for (op_id, prec) in self.op_preconditions.iter().enumerate() {
            scratch.op_remaining_preconditions[op_id] = prec.len() as i32;
        }
        // Empty-precondition operators fire at layer 0 — provided their
        // state-dependent preconditions allow it.
        for (op_id, prec) in self.op_preconditions.iter().enumerate() {
            if prec.is_empty() && scratch.op_eligible[op_id] {
                self.fire_operator(op_id, 0, scratch);
            }
        }
        if self.goal_satisfied(scratch) {
            return Ok(self.goal_max_layer(scratch));
        }

        // Main BFS loop. Ineligible operators never fire — their
        // remaining-precondition counter is never decremented and they
        // can't be triggered through the consumer index. (The
        // counters were initialized above for every op, including
        // ineligibles; the eligibility check here is cheap and keeps the
        // ineligibles' state untouched.)
        while let Some(fid) = scratch.queue.pop_front() {
            let fact_layer = scratch.fact_first_layer[fid];
            for &op_id in &self.consumers[fid] {
                if !scratch.op_eligible[op_id] {
                    continue;
                }
                let remaining = &mut scratch.op_remaining_preconditions[op_id];
                if *remaining > 0 {
                    *remaining -= 1;
                    if *remaining == 0 {
                        self.fire_operator(op_id, fact_layer + 1, scratch);
                    }
                }
            }
            if self.goal_satisfied(scratch) {
                return Ok(self.goal_max_layer(scratch));
            }
        }

        if self.goal_satisfied(scratch) {
            Ok(self.goal_max_layer(scratch))
        } else {
            Ok(i32::MAX)
        }
    }

    fn fire_operator(&self, op_id: OpId, layer: i32, scratch: &mut ScratchBuffers) {
        if scratch.op_first_layer[op_id] >= 0 {
            return;
        }
        scratch.op_first_layer[op_id] = layer;

        // Propositional adds.
        for &fid in &self.op_effects[op_id] {
            if scratch.fact_first_layer[fid] < 0 {
                scratch.fact_first_layer[fid] = layer;
                scratch.queue.push_back(fid);
            }
        }

        let numeric_effects = &self.op_numeric_effects[op_id];
        if numeric_effects.is_empty() {
            // Skip the dirty-var / dirty-axiom plumbing entirely — the
            // common case for purely-propositional operators.
            return;
        }

        // Numeric effects → assignment-axiom propagation → comparison-
        // axiom re-evaluation. Uses preallocated scratch Vec+bitset for
        // dirty-variable / dirty-axiom dedup rather than per-firing
        // HashSet allocations.
        scratch.dirty_vars_scratch.clear();
        scratch.dirty_axioms_scratch.clear();
        for eff in numeric_effects {
            if self.apply_numeric_effect(eff, &mut scratch.numeric)
                && !scratch.dirty_var_mark[eff.affected_var]
            {
                scratch.dirty_var_mark[eff.affected_var] = true;
                scratch.dirty_vars_scratch.push(eff.affected_var);
            }
        }
        if scratch.dirty_vars_scratch.is_empty() {
            return; // numeric envelope didn't actually widen
        }
        // Propagate through assignment axioms; any newly-affected
        // numeric var joins the dirty list so we don't miss its axioms.
        // Re-use the same dirty Vec / mark — propagate_assignment_axioms
        // appends and dedups against `dirty_var_mark`.
        let _ = self.propagate_assignment_axioms(
            &mut scratch.numeric,
            &mut scratch.dirty_vars_scratch,
            &mut scratch.dirty_var_mark,
        );
        // Collect distinct comparison-axiom indices touched.
        for &var in &scratch.dirty_vars_scratch {
            for &ax in &self.axioms_touching_var[var] {
                if !scratch.dirty_axiom_mark[ax] {
                    scratch.dirty_axiom_mark[ax] = true;
                    scratch.dirty_axioms_scratch.push(ax);
                }
            }
        }
        // Re-evaluate and emit; reset marks per axiom as we go so the
        // scratch buffers stay clean for the next firing.
        for i in 0..scratch.dirty_axioms_scratch.len() {
            let axiom_idx = scratch.dirty_axioms_scratch[i];
            if scratch.axiom_first_layer[axiom_idx] < 0 {
                let axiom = &self.comparison_axioms[axiom_idx];
                if self.evaluate_axiom(axiom, &scratch.numeric) {
                    scratch.axiom_first_layer[axiom_idx] = layer;
                    if scratch.fact_first_layer[axiom.true_fact] < 0 {
                        scratch.fact_first_layer[axiom.true_fact] = layer;
                        scratch.queue.push_back(axiom.true_fact);
                    }
                }
            }
        }
        // Reset the marks we set this firing.
        for &var in &scratch.dirty_vars_scratch {
            scratch.dirty_var_mark[var] = false;
        }
        for &ax in &scratch.dirty_axioms_scratch {
            scratch.dirty_axiom_mark[ax] = false;
        }
    }

    fn goal_satisfied(&self, scratch: &ScratchBuffers) -> bool {
        self.goal_facts
            .iter()
            .all(|&gid| scratch.fact_first_layer[gid] >= 0)
    }

    fn goal_max_layer(&self, scratch: &ScratchBuffers) -> i32 {
        self.goal_facts
            .iter()
            .map(|&gid| scratch.fact_first_layer[gid])
            .max()
            .unwrap_or(0)
    }

    fn extract_relaxed_plan(&self, scratch: &mut ScratchBuffers) -> f64 {
        let max_layer = self.goal_max_layer(scratch);
        if max_layer < 0 {
            return 0.0;
        }
        scratch.goals_at_layer.clear();
        scratch
            .goals_at_layer
            .resize((max_layer + 1) as usize, Vec::new());
        for v in &mut scratch.seen {
            *v = false;
        }
        for v in &mut scratch.in_plan {
            *v = false;
        }

        for &gid in &self.goal_facts {
            let layer = scratch.fact_first_layer[gid];
            if layer <= 0 {
                scratch.seen[gid] = true;
                continue;
            }
            if !scratch.seen[gid] {
                scratch.seen[gid] = true;
                scratch.goals_at_layer[layer as usize].push(gid);
            }
        }

        let mut plan_cost = 0.0;
        for layer in (1..=max_layer).rev() {
            let goals_here = std::mem::take(&mut scratch.goals_at_layer[layer as usize]);
            for fid in goals_here {
                // `fire_operator` writes effects at `op_first_layer` — i.e.
                // operator and effect share a layer in this single-counter
                // convention (an op whose last precondition is at fact
                // layer `L` fires at `L+1` and its effects appear at
                // `L+1`). So an op that achieves `fid` at fact layer
                // `layer` has `op_first_layer == layer`, not `layer - 1`.
                let target_op_layer = layer;
                let mut best_op: Option<OpId> = None;
                let mut best_cost = f64::INFINITY;
                for &op_id in &self.achievers[fid] {
                    let op_layer = scratch.op_first_layer[op_id];
                    if op_layer < 0 || op_layer > target_op_layer {
                        continue;
                    }
                    if !scratch.op_eligible[op_id] {
                        continue;
                    }
                    // Effective cost for plan-picking: synthetic ops are
                    // free *given* their parent, but charging the parent
                    // here when not already in the plan is what FF does to
                    // avoid the "free synthetic" loophole. Tie-breaking
                    // still prefers the literally-cheapest op.
                    let effective_cost = if let Some(parent) = self.op_parent[op_id]
                        && !scratch.in_plan[parent]
                    {
                        self.op_cost[parent]
                    } else {
                        self.op_cost[op_id]
                    };
                    if effective_cost < best_cost {
                        best_cost = effective_cost;
                        best_op = Some(op_id);
                    }
                }
                let Some(op_id) = best_op else {
                    continue;
                };
                if scratch.in_plan[op_id] {
                    continue;
                }
                scratch.in_plan[op_id] = true;
                plan_cost += self.op_cost[op_id];
                // Synthetic ops pull their parent in for cost accounting.
                if let Some(parent) = self.op_parent[op_id]
                    && !scratch.in_plan[parent]
                {
                    scratch.in_plan[parent] = true;
                    plan_cost += self.op_cost[parent];
                }
                for &pre_fid in &self.op_preconditions[op_id] {
                    if scratch.seen[pre_fid] {
                        continue;
                    }
                    let pre_layer = scratch.fact_first_layer[pre_fid];
                    if pre_layer <= 0 {
                        scratch.seen[pre_fid] = true;
                        continue;
                    }
                    scratch.seen[pre_fid] = true;
                    if (pre_layer as usize) < scratch.goals_at_layer.len() {
                        scratch.goals_at_layer[pre_layer as usize].push(pre_fid);
                    }
                }
            }
        }
        plan_cost
    }

    /// Task-operator indices of "helpful actions" — operators in the
    /// extracted relaxed plan that are *applicable in the current
    /// concrete state*. These are the operators the search engine
    /// should preferentially try next.
    ///
    /// In this layer convention an op is applicable-now iff all its
    /// preconditions are at fact layer 0 (the initial-fact layer), which
    /// means the op fires at layer 1. Zero-precondition ops fire at
    /// layer 0 — also applicable. So we accept `op_first_layer ∈ {0, 1}`.
    ///
    /// We restrict to operators with a `task_idx` so callers see real
    /// task operators, not the synthetic conditional-effect or
    /// propositional-axiom pseudo-ops. Synthetics that appear in the
    /// plan implicitly pull their parent into the plan (via the
    /// in-extraction `op_parent` accounting), so the parent's
    /// `op_task_idx` is what surfaces here.
    fn collect_helpful_action_ids(&self, scratch: &ScratchBuffers) -> Vec<usize> {
        let mut out = Vec::new();
        for op_id in 0..self.op_preconditions.len() {
            if !scratch.in_plan[op_id] {
                continue;
            }
            let op_layer = scratch.op_first_layer[op_id];
            if !(0..=1).contains(&op_layer) {
                continue;
            }
            if let Some(task_idx) = self.op_task_idx[op_id] {
                out.push(task_idx);
            }
        }
        out
    }
}

/// Transitive numeric-var dependency closure. `depends_on[v]` lists every
/// numeric var `u` such that `v`'s value is computed (directly or
/// recursively) from `u` via `Sum`/`Difference` assignment axioms.
///
/// `v` always depends on itself. The closure converges in at most
/// `num_numeric` iterations because each iteration that adds *anything*
/// strictly grows at least one `depends_on[v]` set, and each set is
/// bounded by `num_numeric`.
fn compute_numeric_dependency_closure(
    num_numeric: usize,
    assignment_axioms: &[AssignmentAxiomDesc],
) -> Vec<Vec<NumVarId>> {
    let mut sets: Vec<HashSet<NumVarId>> = (0..num_numeric)
        .map(|v| {
            let mut s = HashSet::new();
            s.insert(v);
            s
        })
        .collect();
    loop {
        let mut changed = false;
        for ax in assignment_axioms {
            // Snapshot the operands' current closures, then union into
            // the affected var. The clone is intentional — borrowing
            // `sets` mutably for the destination while also reading from
            // it for the sources would need split-borrow gymnastics that
            // aren't worth it for a one-shot construction step.
            let left = sets[ax.left_var].clone();
            let right = sets[ax.right_var].clone();
            let dst = &mut sets[ax.affected_var];
            for v in left.into_iter().chain(right) {
                if dst.insert(v) {
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    sets.into_iter().map(|s| s.into_iter().collect()).collect()
}

/// Does the direction in which `affected_var`'s envelope can move
/// (under an effect with `direction`) advance the satisfaction of
/// `axiom`?
fn axiom_needs_direction(
    affected_var: NumVarId,
    direction: EffectDirection,
    axiom: &ComparisonAxiomDesc,
) -> bool {
    let lhs = axiom.left_var == affected_var;
    let rhs = axiom.right_var == affected_var;
    if !lhs && !rhs {
        return false;
    }
    match axiom.op {
        ComparisonOperator::GreaterThan | ComparisonOperator::GreaterThanOrEqual => {
            // need max[L] big or min[R] small
            (lhs && direction.includes_grow_max()) || (rhs && direction.includes_shrink_min())
        }
        ComparisonOperator::LessThan | ComparisonOperator::LessThanOrEqual => {
            // need min[L] small or max[R] big
            (lhs && direction.includes_shrink_min()) || (rhs && direction.includes_grow_max())
        }
        ComparisonOperator::Equal => {
            // any envelope movement on either side can help meet equality
            true
        }
        ComparisonOperator::UnEqual => {
            // any movement breaks equality
            true
        }
    }
}

impl<'task> Heuristic for FfHeuristic<'task> {
    fn dead_ends_are_reliable(&self) -> bool {
        false
    }

    fn compute_heuristic(
        &self,
        eval_state: &EvaluationState<'_, '_>,
    ) -> Result<f64, EvaluationError> {
        if eval_state.is_goal() {
            self.last_helpful_action_ids.borrow_mut().clear();
            return Ok(0.0);
        }
        let registry = eval_state.state_registry().ok_or_else(|| {
            EvaluationError::ComputationFailed(
                "FF heuristic requires StateRegistry-backed EvaluationState".to_string(),
            )
        })?;
        let mut scratch = self.scratch.borrow_mut();
        scratch.reset();
        let goal_layer = self.build_rpg(eval_state, registry, &mut scratch)?;
        if goal_layer == i32::MAX {
            self.last_helpful_action_ids.borrow_mut().clear();
            return Err(EvaluationError::DeadEnd { reliable: false });
        }
        if goal_layer == 0 {
            self.last_helpful_action_ids.borrow_mut().clear();
            return Ok(0.0);
        }
        let cost = self.extract_relaxed_plan(&mut scratch);
        // Snapshot helpful-action IDs for the get_preferred_operator_ids
        // call the search engine will issue immediately after this returns.
        *self.last_helpful_action_ids.borrow_mut() = self.collect_helpful_action_ids(&scratch);
        Ok(cost)
    }

    fn get_preferred_operators(
        &self,
        _state: &planforge_sas::state_registry::ConcreteState,
    ) -> Vec<planforge_sas::numeric_task::Operator> {
        // The search engine is expected to call `compute_heuristic` for a
        // state before asking for its preferred operators; we serve the
        // snapshot from there. If the engine queries without an
        // intervening `compute_heuristic`, the snapshot is stale — but
        // that's a contract violation, not a fallback.
        let ids = self.last_helpful_action_ids.borrow();
        let task_ops = self.task.get_operators();
        ids.iter()
            .filter_map(|&task_idx| task_ops.get(task_idx).cloned())
            .collect()
    }

    fn get_preferred_operator_ids(&self) -> Vec<usize> {
        self.last_helpful_action_ids.borrow().clone()
    }

    fn heuristic_name(&self) -> &str {
        "ff"
    }
}
