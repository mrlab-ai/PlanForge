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

use crate::numeric::evaluation::evaluator::{EvaluationError, EvaluationState};
use crate::numeric::evaluation::heuristic::Heuristic;
use planforge_sas::numeric::axioms::{CalOperator, ComparisonOperator};
use planforge_sas::numeric::numeric_task::{
    AbstractNumericTask, AssignmentOperation, ExplicitFact, NumericType,
    metric_operator_cost_from_initial_values,
};
use planforge_sas::numeric::state_registry::StateRegistry;

type FactId = usize;
type OpId = usize;
type NumVarId = usize;
type AxiomIdx = usize;

const COMPARISON_TRUE_VALUE: usize = 0;

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
        let new_max = if other.max > self.max { other.max } else { self.max };
        let new_min = if other.min < self.min { other.min } else { self.min };
        // Use bit-pattern inequality rather than `> self.max + EPSILON` so
        // `+∞ vs finite max` reads as "widened" without an arithmetic-on-
        // infinity ambiguity.
        let changed = new_max.to_bits() != self.max.to_bits()
            || new_min.to_bits() != self.min.to_bits();
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
    /// Cache of the most recently extracted helpful-action set. Populated
    /// at the end of every `compute_heuristic`; returned by
    /// `get_preferred_operators`. We don't key by state-id because the
    /// search engine guarantees `compute_heuristic` is invoked on a state
    /// before `get_preferred_operators` is asked about it, and the
    /// `EvaluationState` carries the same state pointer through both
    /// calls — so the cache is fresh for the only call pattern that
    /// matters.
    last_helpful_actions: RefCell<Vec<planforge_sas::numeric::numeric_task::Operator>>,
}

impl<'task> FfHeuristic<'task> {
    pub fn new(task: &'task dyn AbstractNumericTask) -> Result<Self, String> {
        // 1. Propositional variables that are *driven* by comparison
        //    axioms. The axiom's TRUE value is added back to the FF fact
        //    universe in step 2 below; the FALSE / UNKNOWN values are
        //    dropped (the delete relaxation can only ever gain facts).
        //
        //    `AssignmentAxiom::get_affected_var_id` lives in the *numeric*
        //    index namespace — it identifies a numeric variable whose
        //    value is computed from others by the axiom, not a
        //    propositional variable. Do not feed those indices into the
        //    propositional bucket; conflating the namespaces silently
        //    dropped legitimate prop facts in earlier versions.
        let mut comparison_axiom_prop_vars: HashSet<usize> = HashSet::new();
        for axiom in task.comparison_axioms() {
            comparison_axiom_prop_vars.insert(axiom.get_affected_var_id());
        }

        // 2. Enumerate propositional facts (one FactId per non-axiom-var
        //    value) then comparison-axiom TRUE facts (one FactId per axiom).
        let num_props = task.variables().len();
        let mut fact_id_table: Vec<Vec<Option<FactId>>> = (0..num_props)
            .map(|var_id| vec![None; task.variables()[var_id].domain_size()])
            .collect();
        let mut fact_var_value: Vec<(usize, usize)> = Vec::new();
        let mut fact_to_axiom: Vec<Option<AxiomIdx>> = Vec::new();

        for var_id in 0..num_props {
            if comparison_axiom_prop_vars.contains(&var_id) {
                // Skip — only the TRUE value (registered in step 3) is
                // representable under the monotonic relaxation.
                continue;
            }
            for value in 0..task.variables()[var_id].domain_size() {
                let fid = fact_var_value.len();
                fact_id_table[var_id][value] = Some(fid);
                fact_var_value.push((var_id, value));
                fact_to_axiom.push(None);
            }
        }

        let mut comparison_axioms = Vec::with_capacity(task.comparison_axioms().len());
        for (axiom_idx, axiom) in task.comparison_axioms().iter().enumerate() {
            let fid = fact_var_value.len();
            let affected = axiom.get_affected_var_id();
            if affected >= num_props {
                return Err(format!(
                    "comparison axiom {axiom_idx} affects out-of-range variable {affected}"
                ));
            }
            let row = &mut fact_id_table[affected];
            if row.is_empty() {
                *row = vec![None; task.variables()[affected].domain_size()];
            }
            if COMPARISON_TRUE_VALUE >= row.len() {
                return Err(format!(
                    "comparison axiom {axiom_idx} affected variable has no TRUE value"
                ));
            }
            row[COMPARISON_TRUE_VALUE] = Some(fid);
            fact_var_value.push((affected, COMPARISON_TRUE_VALUE));
            fact_to_axiom.push(Some(axiom_idx));
            comparison_axioms.push(ComparisonAxiomDesc {
                true_fact: fid,
                left_var: axiom.get_left_var_id(),
                right_var: axiom.get_right_var_id(),
                op: axiom.get_operator().clone(),
            });
        }
        let num_facts = fact_var_value.len();

        let map_fact = |fact: &ExplicitFact| -> Option<FactId> {
            if fact.var() >= fact_id_table.len() {
                return None;
            }
            let row = &fact_id_table[fact.var()];
            if fact.value() >= row.len() {
                return None;
            }
            row[fact.value()]
        };

        // 3. Axiom-by-var index.
        let num_numeric = task.numeric_variables().len();
        let mut axioms_touching_var: Vec<Vec<AxiomIdx>> = vec![Vec::new(); num_numeric];
        for (idx, ax) in comparison_axioms.iter().enumerate() {
            if ax.left_var >= num_numeric || ax.right_var >= num_numeric {
                return Err(format!(
                    "comparison axiom {idx} references out-of-range numeric variable \
                     (left={}, right={}, num_numeric={num_numeric})",
                    ax.left_var, ax.right_var
                ));
            }
            axioms_touching_var[ax.left_var].push(idx);
            if ax.right_var != ax.left_var {
                axioms_touching_var[ax.right_var].push(idx);
            }
        }

        // 3b. Assignment axioms. Each computes a derived numeric value from
        //     two operand numerics; we'll re-propagate bounds through
        //     these during the RPG forward pass.
        let mut assignment_axioms: Vec<AssignmentAxiomDesc> = Vec::new();
        for (axiom_idx, axiom) in task.assignment_axioms().iter().enumerate() {
            let affected = axiom.get_affected_var_id();
            let left = axiom.get_left_var_id();
            let right = axiom.get_right_var_id();
            if affected >= num_numeric || left >= num_numeric || right >= num_numeric {
                return Err(format!(
                    "assignment axiom {axiom_idx} references out-of-range numeric variable \
                     (affected={affected}, left={left}, right={right}, num_numeric={num_numeric})"
                ));
            }
            match axiom.get_operator() {
                CalOperator::Sum | CalOperator::Difference => {
                    assignment_axioms.push(AssignmentAxiomDesc {
                        affected_var: affected,
                        left_var: left,
                        right_var: right,
                        op: axiom.get_operator().clone(),
                    });
                }
                CalOperator::Product | CalOperator::Division => {
                    return Err(format!(
                        "assignment axiom {axiom_idx} uses unsupported {:?} operator. \
                         Monotonic-relaxation bounds for multiplicative derived numerics \
                         require sign-aware case analysis which this FF doesn't implement. \
                         Pick a different heuristic for such tasks.",
                        axiom.get_operator()
                    ));
                }
            }
        }

        // 4. Capture each Constant numeric variable's initial value so we
        //    can classify effect directions at construction time.
        let initial_numeric = task.get_initial_numeric_state_values();
        let constant_value: Vec<Option<f64>> = task
            .numeric_variables()
            .iter()
            .enumerate()
            .map(|(idx, var)| match var.get_type() {
                NumericType::Constant => Some(initial_numeric.get(idx).copied().ok_or_else(
                    || format!("constant numeric variable {idx} missing initial value"),
                )),
                _ => None,
            })
            .map(|opt| opt.transpose())
            .collect::<Result<_, _>>()?;
        drop(initial_numeric);

        let direction_of_effect = |op: &AssignmentOperation, rhs: NumVarId| -> EffectDirection {
            // For Constant RHS the direction is exact. For non-constant
            // RHS we cannot determine signs statically — the envelope is
            // assumed bidirectional, which conservatively widens the
            // achiever set without losing soundness.
            let rhs_const = constant_value.get(rhs).copied().flatten();
            match op {
                AssignmentOperation::Plus => match rhs_const {
                    Some(v) if v > 0.0 => EffectDirection::GrowMax,
                    Some(v) if v < 0.0 => EffectDirection::ShrinkMin,
                    Some(_) => EffectDirection::Both, // exact zero — no movement
                    None => EffectDirection::Both,
                },
                AssignmentOperation::Minus => match rhs_const {
                    Some(v) if v > 0.0 => EffectDirection::ShrinkMin,
                    Some(v) if v < 0.0 => EffectDirection::GrowMax,
                    Some(_) => EffectDirection::Both,
                    None => EffectDirection::Both,
                },
                AssignmentOperation::Assign => EffectDirection::Both,
                AssignmentOperation::Times | AssignmentOperation::Divide => {
                    // These never reach `direction_of_effect`; rejected at
                    // operator-collection time below.
                    EffectDirection::Both
                }
            }
        };

        // 5. Operator collection. Each task operator becomes one "real" op
        //    plus zero or more "synthetic" ops, one per conditional effect.
        let operators = task.get_operators();
        let mut op_preconditions: Vec<Vec<FactId>> = Vec::new();
        let mut op_effects: Vec<Vec<FactId>> = Vec::new();
        let mut op_numeric_effects: Vec<Vec<AssignmentEffectDesc>> = Vec::new();
        let mut op_cost: Vec<f64> = Vec::new();
        let mut op_parent: Vec<Option<OpId>> = Vec::new();

        // Each parent operator may have state-dependent preconditions (e.g.
        // require a comparison-axiom FALSE value); those are checked
        // against the live state at evaluation time. Both the parent op
        // and any synthetic conditional-effect ops derived from it inherit
        // the parent's state-dependent preconds.
        let mut op_state_deps: Vec<Vec<StateDependentPrecond>> = Vec::new();
        let mut op_task_idx: Vec<Option<usize>> = Vec::new();

        for (op_idx, op) in operators.iter().enumerate() {
            let mut parent_preconds: Vec<FactId> = Vec::new();
            let mut parent_state_deps: Vec<StateDependentPrecond> = Vec::new();
            for pre in op.preconditions() {
                match map_fact(pre) {
                    Some(fid) => parent_preconds.push(fid),
                    None => parent_state_deps.push((pre.var(), pre.value())),
                }
            }
            let parent_op_id = op_preconditions.len();

            // Parent op: unconditional propositional and numeric effects.
            let mut parent_effects: Vec<FactId> = Vec::new();
            for eff in op.effects() {
                if !eff.conditions().is_empty() {
                    continue;
                }
                if let Some(fid) = map_fact(&ExplicitFact::new(eff.var_id(), eff.value())) {
                    parent_effects.push(fid);
                }
            }
            let mut parent_numeric: Vec<AssignmentEffectDesc> = Vec::new();
            for assign in op.assignment_effects() {
                if !assign.conditions().is_empty() {
                    continue;
                }
                match assign.operation() {
                    AssignmentOperation::Plus
                    | AssignmentOperation::Minus
                    | AssignmentOperation::Assign => {
                        parent_numeric.push(AssignmentEffectDesc {
                            affected_var: assign.affected_var_id(),
                            operation: assign.operation().clone(),
                            rhs_var: assign.var_id(),
                            direction: direction_of_effect(
                                assign.operation(),
                                assign.var_id(),
                            ),
                        });
                    }
                    AssignmentOperation::Times | AssignmentOperation::Divide => {
                        return Err(format!(
                            "operator {op_idx} (`{}`) uses unsupported {:?} assignment effect; \
                             the monotonic relaxation can't soundly bound it. Pick a different \
                             heuristic for tasks that need multiplicative numerics.",
                            op.name(),
                            assign.operation()
                        ));
                    }
                }
            }
            let parent_cost =
                metric_operator_cost_from_initial_values(task, op).max(0.0);
            op_preconditions.push(parent_preconds.clone());
            op_state_deps.push(parent_state_deps.clone());
            op_effects.push(parent_effects);
            op_numeric_effects.push(parent_numeric);
            op_cost.push(parent_cost);
            op_parent.push(None);
            op_task_idx.push(Some(op_idx));

            // Synthetic ops: one per conditional effect. Cost is 0; parent
            // is the real op above. Preconditions union the parent's
            // mappable preconds with the conditional effect's own
            // conditions; the parent's state-dependent preconds are
            // inherited verbatim; the conditional effect's own conditions
            // are split similarly into mappable / state-dependent.
            for eff in op.effects() {
                if eff.conditions().is_empty() {
                    continue;
                }
                let mut precs = parent_preconds.clone();
                let mut state_deps = parent_state_deps.clone();
                for cond in eff.conditions() {
                    match map_fact(cond) {
                        Some(fid) => precs.push(fid),
                        None => state_deps.push((cond.var(), cond.value())),
                    }
                }
                let mut effs = Vec::new();
                if let Some(fid) = map_fact(&ExplicitFact::new(eff.var_id(), eff.value())) {
                    effs.push(fid);
                }
                op_preconditions.push(precs);
                op_state_deps.push(state_deps);
                op_effects.push(effs);
                op_numeric_effects.push(Vec::new());
                op_cost.push(0.0);
                op_parent.push(Some(parent_op_id));
                op_task_idx.push(None);
            }
            for assign in op.assignment_effects() {
                if assign.conditions().is_empty() {
                    continue;
                }
                let mut precs = parent_preconds.clone();
                let mut state_deps = parent_state_deps.clone();
                for cond in assign.conditions() {
                    match map_fact(cond) {
                        Some(fid) => precs.push(fid),
                        None => state_deps.push((cond.var(), cond.value())),
                    }
                }
                let numeric = match assign.operation() {
                    AssignmentOperation::Plus
                    | AssignmentOperation::Minus
                    | AssignmentOperation::Assign => {
                        vec![AssignmentEffectDesc {
                            affected_var: assign.affected_var_id(),
                            operation: assign.operation().clone(),
                            rhs_var: assign.var_id(),
                            direction: direction_of_effect(
                                assign.operation(),
                                assign.var_id(),
                            ),
                        }]
                    }
                    AssignmentOperation::Times | AssignmentOperation::Divide => {
                        return Err(format!(
                            "operator {op_idx} (`{}`) uses unsupported {:?} conditional \
                             assignment effect.",
                            op.name(),
                            assign.operation()
                        ));
                    }
                };
                op_preconditions.push(precs);
                op_state_deps.push(state_deps);
                op_effects.push(Vec::new());
                op_numeric_effects.push(numeric);
                op_cost.push(0.0);
                op_parent.push(Some(parent_op_id));
                op_task_idx.push(None);
            }
        }

        // 5b. Propositional axioms (`task.axioms()`) — these derive a fact
        //     (var_id, effect_value) when their `conditions` hold, with no
        //     cost. Modeled as a zero-cost pseudo-operator. Both
        //     `precondition_value` and `effect_value` matter: the axiom
        //     fires for the var-value transition `precondition_value →
        //     effect_value` once `conditions` are reached. Under the
        //     monotonic relaxation we add the effect once.
        for (axiom_idx, axiom) in task.axioms().iter().enumerate() {
            // Effect: map `(var_id, effect_value)` to a FactId. Out-of-
            // universe effects mean the axiom drives a value of a
            // numeric-axiom variable, which the relaxation cannot
            // represent; fail loudly rather than silently drop.
            let Some(effect_fid) =
                map_fact(&ExplicitFact::new(axiom.var_id(), axiom.effect_value()))
            else {
                return Err(format!(
                    "propositional axiom {axiom_idx} effect on \
                     variable {} value {} is unrepresentable in the FF \
                     fact universe (likely a numeric-axiom-driven variable)",
                    axiom.var_id(),
                    axiom.effect_value()
                ));
            };
            // Preconditions: the axiom's `conditions` *plus* the
            // precondition-value assumption on the affected variable
            // itself. Each precondition is split between the FF universe
            // (`FactId`) and the state-dependent escape hatch (e.g. the
            // axiom's pre-value is `UNKNOWN` which monotonic relaxation
            // can't fabricate — check live at evaluation time).
            let mut precs: Vec<FactId> = Vec::new();
            let mut state_deps: Vec<StateDependentPrecond> = Vec::new();
            for cond in axiom.conditions() {
                match map_fact(cond) {
                    Some(fid) => precs.push(fid),
                    None => state_deps.push((cond.var(), cond.value())),
                }
            }
            let pre_value_fact =
                ExplicitFact::new(axiom.var_id(), axiom.precondition_value());
            match map_fact(&pre_value_fact) {
                Some(prec_fid) => precs.push(prec_fid),
                None => state_deps.push((axiom.var_id(), axiom.precondition_value())),
            }
            op_preconditions.push(precs);
            op_state_deps.push(state_deps);
            op_effects.push(vec![effect_fid]);
            op_numeric_effects.push(Vec::new());
            op_cost.push(0.0);
            op_parent.push(None);
            op_task_idx.push(None);
        }

        let num_ops = op_preconditions.len();

        // 6. Build achiever index. Propositional add-effects: straightforward
        //    op → fact mapping. Comparison-axiom TRUE facts: only ops whose
        //    numeric effects can push the envelope in the right direction.
        let mut achievers: Vec<Vec<OpId>> = vec![Vec::new(); num_facts];
        for (op_id, effs) in op_effects.iter().enumerate() {
            for &fid in effs {
                achievers[fid].push(op_id);
            }
        }
        for (op_id, numeric_effs) in op_numeric_effects.iter().enumerate() {
            for eff in numeric_effs {
                if eff.affected_var >= num_numeric {
                    return Err(format!(
                        "operator {op_id} effect on out-of-range numeric variable {}",
                        eff.affected_var
                    ));
                }
                for &axiom_idx in &axioms_touching_var[eff.affected_var] {
                    let axiom = &comparison_axioms[axiom_idx];
                    if !axiom_needs_direction(
                        eff.affected_var,
                        eff.direction,
                        axiom,
                    ) {
                        continue;
                    }
                    let true_fact = axiom.true_fact;
                    if achievers[true_fact].last() != Some(&op_id) {
                        // dedup against the most recent insertion only —
                        // operator-with-multiple-effects-on-same-axiom would
                        // otherwise repeat itself.
                        if !achievers[true_fact].contains(&op_id) {
                            achievers[true_fact].push(op_id);
                        }
                    }
                }
            }
        }

        // 7. Goals.
        let goal_facts: Vec<FactId> = (0..task.get_num_goals())
            .map(|i| {
                let goal = task.get_goal_fact(i);
                map_fact(goal).ok_or_else(|| {
                    format!(
                        "goal fact {goal:?} maps to no FactId — variable {} value {} not \
                         in the FF fact universe (numeric-axiom non-TRUE goals are not \
                         representable under the delete relaxation)",
                        goal.var(),
                        goal.value()
                    )
                })
            })
            .collect::<Result<_, _>>()?;

        // 8. Consumer index for the BFS.
        let mut consumers: Vec<Vec<OpId>> = vec![Vec::new(); num_facts];
        for (op_id, prec) in op_preconditions.iter().enumerate() {
            for &fid in prec {
                consumers[fid].push(op_id);
            }
        }

        let num_axioms_for_scratch = comparison_axioms.len();
        Ok(Self {
            task,
            op_task_idx,
            op_preconditions,
            op_state_deps,
            op_effects,
            op_numeric_effects,
            op_cost,
            op_parent,
            goal_facts,
            achievers,
            consumers,
            fact_var_value,
            fact_to_axiom,
            comparison_axioms,
            assignment_axioms,
            axioms_touching_var,
            num_facts,
            num_numeric,
            scratch: RefCell::new(ScratchBuffers::new(
                num_facts,
                num_ops,
                num_numeric,
                num_axioms_for_scratch,
            )),
            last_helpful_actions: RefCell::new(Vec::new()),
        })
    }

    fn state_holds_fact(
        &self,
        state: &planforge_sas::numeric::state_registry::ConcreteState,
        registry: &StateRegistry<'_>,
        fid: FactId,
    ) -> bool {
        let (var, value) = self.fact_var_value[fid];
        let fact = ExplicitFact::new(var, value);
        fact.is_hold(state, registry)
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
    /// catch dependencies among derived vars. Returns the set of numeric
    /// vars whose range was widened by this pass (useful so comparison-
    /// axiom re-evaluation knows what to re-check).
    fn propagate_assignment_axioms(
        &self,
        numeric: &mut [NumericRange],
    ) -> HashSet<NumVarId> {
        let mut dirty: HashSet<NumVarId> = HashSet::new();
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
                    CalOperator::Product | CalOperator::Division => {
                        unreachable!(
                            "Product/Division assignment axioms should have been rejected \
                             at construction"
                        );
                    }
                };
                if numeric[ax.affected_var].join(new) {
                    dirty.insert(ax.affected_var);
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        dirty
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
            ComparisonOperator::UnEqual => {
                l.min != l.max || r.min != r.max || l.min != r.min
            }
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
        scratch.op_eligible.resize(self.op_preconditions.len(), true);
        let live_state = eval_state.state();
        for (op_id, deps) in self.op_state_deps.iter().enumerate() {
            if deps.is_empty() {
                continue;
            }
            let eligible = deps.iter().all(|&(var, value)| {
                ExplicitFact::new(var, value).is_hold(live_state, registry)
            });
            scratch.op_eligible[op_id] = eligible;
        }

        scratch.numeric = self.initial_numeric_state(eval_state, registry)?;
        // The initial state already evaluates derived numerics correctly,
        // but `fill_numeric_vars` returns singleton ranges for them. Run
        // assignment-axiom propagation once so any wider-than-singleton
        // bounds (e.g. uninitialized derived = -∞/+∞) settle to a
        // consistent starting point.
        let _ = self.propagate_assignment_axioms(&mut scratch.numeric);

        // Layer 0 propositional facts.
        for fid in 0..self.num_facts {
            if self.fact_to_axiom[fid].is_some() {
                continue;
            }
            if self.state_holds_fact(eval_state.state(), registry, fid) {
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

        // Numeric effects → assignment-axiom propagation → comparison-
        // axiom re-evaluation.
        let mut dirty_vars: HashSet<NumVarId> = HashSet::new();
        for eff in &self.op_numeric_effects[op_id] {
            if self.apply_numeric_effect(eff, &mut scratch.numeric) {
                dirty_vars.insert(eff.affected_var);
            }
        }
        if !dirty_vars.is_empty() {
            // Push the change through any assignment axioms; their
            // affected variables join `dirty_vars` so the comparison-
            // axiom touch index picks them up too.
            let derived_changes = self.propagate_assignment_axioms(&mut scratch.numeric);
            dirty_vars.extend(derived_changes);
        }
        let mut dirty_axioms: HashSet<AxiomIdx> = HashSet::new();
        for var in &dirty_vars {
            for &ax in &self.axioms_touching_var[*var] {
                dirty_axioms.insert(ax);
            }
        }
        for axiom_idx in dirty_axioms {
            if scratch.axiom_first_layer[axiom_idx] >= 0 {
                continue;
            }
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
                let target_op_layer = layer - 1;
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

    /// "Helpful actions" — operators in the extracted relaxed plan that
    /// fire at layer 0 (i.e. are *applicable in the current concrete
    /// state*). These are the operators the search engine should
    /// preferentially try next.
    ///
    /// We restrict to operators with a `task_idx` so callers see real
    /// task operators, not the synthetic conditional-effect or
    /// propositional-axiom pseudo-ops. Synthetics that appear in the
    /// plan implicitly pull their parent into the plan (via the
    /// in-extraction `op_parent` accounting), so the parent's
    /// `op_task_idx` is what surfaces here.
    fn collect_helpful_actions(
        &self,
        scratch: &ScratchBuffers,
    ) -> Vec<planforge_sas::numeric::numeric_task::Operator> {
        let mut out = Vec::new();
        let task_ops = self.task.get_operators();
        for op_id in 0..self.op_preconditions.len() {
            if !scratch.in_plan[op_id] {
                continue;
            }
            if scratch.op_first_layer[op_id] != 0 {
                continue;
            }
            let Some(task_idx) = self.op_task_idx[op_id] else {
                continue;
            };
            if let Some(op) = task_ops.get(task_idx) {
                out.push(op.clone());
            }
        }
        out
    }
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
            (lhs && direction.includes_grow_max())
                || (rhs && direction.includes_shrink_min())
        }
        ComparisonOperator::LessThan | ComparisonOperator::LessThanOrEqual => {
            // need min[L] small or max[R] big
            (lhs && direction.includes_shrink_min())
                || (rhs && direction.includes_grow_max())
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
            self.last_helpful_actions.borrow_mut().clear();
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
            self.last_helpful_actions.borrow_mut().clear();
            return Err(EvaluationError::DeadEnd { reliable: false });
        }
        if goal_layer == 0 {
            self.last_helpful_actions.borrow_mut().clear();
            return Ok(0.0);
        }
        let cost = self.extract_relaxed_plan(&mut scratch);
        // Snapshot helpful actions for the get_preferred_operators call
        // the search engine will issue immediately after this returns.
        *self.last_helpful_actions.borrow_mut() = self.collect_helpful_actions(&scratch);
        Ok(cost)
    }

    fn get_preferred_operators(
        &self,
        _state: &planforge_sas::numeric::state_registry::ConcreteState,
    ) -> Vec<planforge_sas::numeric::numeric_task::Operator> {
        // The search engine is expected to call `compute_heuristic` for a
        // state before asking for its preferred operators; we serve the
        // snapshot from there. If the engine queries without an
        // intervening `compute_heuristic`, the snapshot is stale — but
        // that's a contract violation, not a fallback.
        self.last_helpful_actions.borrow().clone()
    }

    fn heuristic_name(&self) -> String {
        "ff".to_string()
    }
}
