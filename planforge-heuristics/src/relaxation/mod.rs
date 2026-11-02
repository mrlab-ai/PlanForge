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

use planforge_sas::axioms::{AssignmentAxiom, NumericExpression};
use planforge_sas::numeric_task::Task;
use planforge_sas::numeric_task::VariableKind;
use planforge_search::heuristic_factory::HeuristicBuildError;
use std::collections::{HashSet, VecDeque};

use planforge_sas::axioms::{CalOperator, ComparisonOperator, PropositionalAxiom};
use planforge_sas::default_value_axioms::{DefaultValueAxiomMode, default_value_axioms};
use planforge_sas::numeric_conditions::ConditionValue;
use planforge_sas::numeric_task::{
    AssignmentEffect, AssignmentOperation, AxiomIndex, ExplicitFact, ExplicitValueIndex, INF_VALUE,
    NEG_INF_VALUE, NumericType, NumericValue, Operator, OperatorIndex, VariableIndex, ZERO_VALUE,
};
use planforge_sas::utils::interval::Interval;
use planforge_search::evaluation::Heuristic;
use planforge_search::evaluation::{Estimate, EvaluationError, EvaluationState};

mod graph;
mod numeric;
use graph::*;
use numeric::*;

type FactId = usize;
type OpId = OperatorIndex;
type NumVarId = VariableIndex;
type AxiomIdx = AxiomIndex;

/// Per-state RPG buffers reused from one evaluation to the next.
#[derive(Default)]
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
    numeric_raw: Vec<NumericValue>,
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
    prop_state_values: Vec<ExplicitValueIndex>,
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
            numeric_raw: Vec::with_capacity(num_numeric),
            numeric: vec![NumericRange::singleton(NumericValue::new(0.0)); num_numeric],
            axiom_first_layer: vec![-1; num_axioms],
            dirty_vars_scratch: Vec::new(),
            dirty_axioms_scratch: Vec::new(),
            dirty_var_mark: vec![false; num_numeric],
            dirty_axiom_mark: vec![false; num_axioms],
            prop_state_values: Vec::new(),
        }
    }

    fn reset(&mut self) {
        self.fact_first_layer.fill(-1);
        self.op_remaining_preconditions.clear();
        self.op_first_layer.fill(-1);
        self.queue.clear();
        for goals in &mut self.goals_at_layer {
            goals.clear();
        }
        self.seen.fill(false);
        self.in_plan.fill(false);
        self.op_eligible.fill(true);
        self.axiom_first_layer.fill(-1);
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

pub struct FfHeuristic {
    /// For each (real or synthetic) operator in `op_preconditions`, the
    /// index into `task.operators()` if it's a real operator (used for
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
    state_dep_ops: Vec<(usize, Box<[StateDependentPrecond]>)>,
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
    fact_id_by_var_value: Vec<Option<FactId>>,
    fact_id_offsets: Vec<usize>,
    /// `fact_to_axiom[fid]` is `Some(axiom_idx)` iff this fact represents
    /// a comparison-axiom TRUE value; `None` for ordinary prop facts.
    fact_to_axiom: Vec<Option<AxiomIdx>>,
    comparison_axioms: Vec<ComparisonAxiomDesc>,
    /// Assignment axioms in topological (SAS axiom-layer) order. Each
    /// describes a derived numeric variable as `affected := left ∘ right`
    /// for `∘ ∈ {Sum, Difference}`.
    assignment_axioms: Vec<AssignmentAxiom>,
    /// For each numeric var, indices of comparison axioms whose LHS or
    /// RHS mentions it. Lets `fire_operator` re-evaluate only the affected
    /// comparison axioms after a numeric update.
    axioms_touching_var: Vec<Vec<AxiomIdx>>,
    num_numeric: usize,
    scratch: ScratchBuffers,
}

impl FfHeuristic {
    /// Compile the task into the relaxed planning graph this heuristic
    /// evaluates: a fact universe, the operators over it, and the achiever and
    /// consumer indices the BFS and relaxed-plan extraction walk.
    pub fn new(task: &Task) -> Result<Self, HeuristicBuildError> {
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
        let state_dep_ops = operators
            .state_deps
            .into_iter()
            .enumerate()
            .filter_map(|(op_id, deps)| {
                (!deps.is_empty()).then(|| (op_id, deps.into_boxed_slice()))
            })
            .collect();
        Ok(Self {
            op_task_idx: operators.task_idx,
            op_preconditions: operators.preconditions,
            state_dep_ops,
            op_effects: operators.effects,
            op_numeric_effects: operators.numeric_effects,
            op_cost: operators.cost,
            op_parent: operators.parent,
            goal_facts,
            achievers,
            consumers,
            fact_id_by_var_value: universe.fact_id_by_var_value,
            fact_id_offsets: universe.fact_id_offsets,
            fact_to_axiom: universe.to_axiom,
            comparison_axioms: universe.comparison_axioms,
            assignment_axioms,
            axioms_touching_var,
            num_numeric,
            scratch: ScratchBuffers::new(num_facts, num_ops, num_numeric, num_comparison_axioms),
        })
    }

    fn fill_initial_numeric_state(
        &self,
        eval_state: &EvaluationState<'_, '_>,
        raw: &mut Vec<NumericValue>,
        ranges: &mut Vec<NumericRange>,
    ) -> Result<(), EvaluationError> {
        eval_state
            .state_registry()
            .view(eval_state.state())
            .fill_numeric(raw)
            .map_err(|err| {
                EvaluationError::ComputationFailed(format!(
                    "FF heuristic failed to read numeric state: {err:?}"
                ))
            })?;
        if raw.len() != self.num_numeric {
            return Err(EvaluationError::ComputationFailed(format!(
                "FF heuristic: numeric-state length ({}) disagrees with task numeric-variable \
                 count ({})",
                raw.len(),
                self.num_numeric
            )));
        }
        ranges.resize(
            self.num_numeric,
            NumericRange::singleton(NumericValue::new(0.0)),
        );
        for (range, &value) in ranges.iter_mut().zip(raw.iter()) {
            *range = NumericRange::singleton(value);
        }
        Ok(())
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
                let new = match &ax.expression {
                    NumericExpression::Linear(terms) => NumericRange::affine(terms, numeric),
                    NumericExpression::Binary {
                        operator,
                        left,
                        right,
                    } => {
                        let l = numeric[left.index()];
                        let r = numeric[right.index()];
                        let lhs = Interval::closed(l.min, l.max);
                        let rhs = Interval::closed(r.min, r.max);
                        let image = match operator {
                            CalOperator::Sum => lhs + rhs,
                            CalOperator::Difference => lhs - rhs,
                            CalOperator::Product => lhs * rhs,
                            CalOperator::Division => lhs / rhs,
                        };
                        NumericRange {
                            min: image.lower,
                            max: image.upper,
                        }
                    }
                };
                if numeric[ax.affected_var_id.index()].join(new) {
                    if !dirty_mark[ax.affected_var_id.index()] {
                        dirty_mark[ax.affected_var_id.index()] = true;
                        dirty_out.push(ax.affected_var_id);
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
        let l = numeric[axiom.left_var.index()];
        let r = numeric[axiom.right_var.index()];
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
        let rhs = numeric[eff.rhs_var.index()];
        let prev = numeric[eff.affected_var.index()];
        let new = match eff.operation {
            AssignmentOperation::Assign => NumericRange {
                max: prev.max.max(rhs.max),
                min: prev.min.min(rhs.min),
            },
            AssignmentOperation::Plus => {
                let mut next = prev;
                if rhs.max > ZERO_VALUE {
                    next.max = INF_VALUE;
                }
                if rhs.min < ZERO_VALUE {
                    next.min = NEG_INF_VALUE;
                }
                next
            }
            AssignmentOperation::Minus => {
                let mut next = prev;
                if rhs.min < ZERO_VALUE {
                    next.max = INF_VALUE;
                }
                if rhs.max > ZERO_VALUE {
                    next.min = NEG_INF_VALUE;
                }
                next
            }
            AssignmentOperation::Times | AssignmentOperation::Divide => {
                unreachable!(
                    "Times/Divide assignment effects should have been rejected at construction"
                );
            }
        };
        numeric[eff.affected_var.index()].join(new)
    }

    fn build_rpg(
        &self,
        eval_state: &EvaluationState<'_, '_>,
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
        let registry = eval_state.state_registry();
        for (op_id, deps) in &self.state_dep_ops {
            let eligible = deps.iter().all(|&(var, value)| {
                ExplicitFact::propositional(var, value).is_held(registry.view(live_state))
            });
            scratch.op_eligible[*op_id] = eligible;
        }

        self.fill_initial_numeric_state(
            eval_state,
            &mut scratch.numeric_raw,
            &mut scratch.numeric,
        )?;
        // The initial state already evaluates derived numerics correctly,
        // but the decoded numeric state gives singleton ranges for them. Run
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
            scratch.dirty_var_mark[v.index()] = false;
        }
        scratch.dirty_vars_scratch.clear();

        // Layer 0 propositional facts. Batch-read the entire packed
        // propositional state once, then index directly into the resulting
        // Vec — saves O(num_facts) bound-checked `state_packer.get` calls
        // per evaluation.
        registry
            .view(live_state)
            .fill_propositional(&mut scratch.prop_state_values);
        for (var, &value) in scratch.prop_state_values.iter().enumerate() {
            let Some(&offset) = self.fact_id_offsets.get(var) else {
                break;
            };
            let end = self
                .fact_id_offsets
                .get(var + 1)
                .copied()
                .unwrap_or(self.fact_id_by_var_value.len());
            if value.index() >= end - offset {
                continue;
            }
            let Some(fid) = self.fact_id_by_var_value[offset + value.index()] else {
                continue;
            };
            if self.fact_to_axiom[fid].is_none() {
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
        scratch.op_remaining_preconditions.clear();
        scratch
            .op_remaining_preconditions
            .reserve(self.op_preconditions.len());
        for (op_id, prec) in self.op_preconditions.iter().enumerate() {
            scratch.op_remaining_preconditions.push(prec.len() as i32);
            if prec.is_empty() && scratch.op_eligible[op_id] {
                self.fire_operator(OperatorIndex::from_usize(op_id), 0, scratch);
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
                if !scratch.op_eligible[op_id.index()] {
                    continue;
                }
                let remaining = &mut scratch.op_remaining_preconditions[op_id.index()];
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
        if scratch.op_first_layer[op_id.index()] >= 0 {
            return;
        }
        scratch.op_first_layer[op_id.index()] = layer;

        // Propositional adds.
        for &fid in &self.op_effects[op_id.index()] {
            if scratch.fact_first_layer[fid] < 0 {
                scratch.fact_first_layer[fid] = layer;
                scratch.queue.push_back(fid);
            }
        }

        let numeric_effects = &self.op_numeric_effects[op_id.index()];
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
                && !scratch.dirty_var_mark[eff.affected_var.index()]
            {
                scratch.dirty_var_mark[eff.affected_var.index()] = true;
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
            for &ax in &self.axioms_touching_var[var.index()] {
                if !scratch.dirty_axiom_mark[ax.index()] {
                    scratch.dirty_axiom_mark[ax.index()] = true;
                    scratch.dirty_axioms_scratch.push(ax);
                }
            }
        }
        // Re-evaluate and emit; reset marks per axiom as we go so the
        // scratch buffers stay clean for the next firing.
        for i in 0..scratch.dirty_axioms_scratch.len() {
            let axiom_idx = scratch.dirty_axioms_scratch[i];
            if scratch.axiom_first_layer[axiom_idx.index()] < 0 {
                let axiom = &self.comparison_axioms[axiom_idx.index()];
                if self.evaluate_axiom(axiom, &scratch.numeric) {
                    scratch.axiom_first_layer[axiom_idx.index()] = layer;
                    if scratch.fact_first_layer[axiom.true_fact] < 0 {
                        scratch.fact_first_layer[axiom.true_fact] = layer;
                        scratch.queue.push_back(axiom.true_fact);
                    }
                }
            }
        }
        // Reset the marks we set this firing.
        for &var in &scratch.dirty_vars_scratch {
            scratch.dirty_var_mark[var.index()] = false;
        }
        for &ax in &scratch.dirty_axioms_scratch {
            scratch.dirty_axiom_mark[ax.index()] = false;
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

    fn extract_relaxed_plan(
        &self,
        scratch: &mut ScratchBuffers,
        mut helpful_action_ids: Option<&mut Vec<OperatorIndex>>,
    ) -> f64 {
        let max_layer = self.goal_max_layer(scratch);
        if max_layer < 0 {
            return 0.0;
        }
        let needed_layers = (max_layer + 1) as usize;
        for goals in &mut scratch.goals_at_layer {
            goals.clear();
        }
        while scratch.goals_at_layer.len() < needed_layers {
            scratch.goals_at_layer.push(Vec::new());
        }
        scratch.seen.fill(false);
        scratch.in_plan.fill(false);

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
            let layer_index = layer as usize;
            let mut goal_index = 0;
            while goal_index < scratch.goals_at_layer[layer_index].len() {
                let fid = scratch.goals_at_layer[layer_index][goal_index];
                goal_index += 1;
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
                    let op_layer = scratch.op_first_layer[op_id.index()];
                    if op_layer < 0 || op_layer > target_op_layer {
                        continue;
                    }
                    if !scratch.op_eligible[op_id.index()] {
                        continue;
                    }
                    // Effective cost for plan-picking: synthetic ops are
                    // free *given* their parent, but charging the parent
                    // here when not already in the plan is what FF does to
                    // avoid the "free synthetic" loophole. Tie-breaking
                    // still prefers the literally-cheapest op.
                    let effective_cost = if let Some(parent) = self.op_parent[op_id.index()]
                        && !scratch.in_plan[parent.index()]
                    {
                        self.op_cost[parent.index()]
                    } else {
                        self.op_cost[op_id.index()]
                    };
                    if effective_cost < best_cost {
                        best_cost = effective_cost;
                        best_op = Some(op_id);
                    }
                }
                let Some(op_id) = best_op else {
                    continue;
                };
                if scratch.in_plan[op_id.index()] {
                    continue;
                }
                scratch.in_plan[op_id.index()] = true;
                if (0..=1).contains(&scratch.op_first_layer[op_id.index()])
                    && let Some(task_idx) = self.op_task_idx[op_id.index()]
                    && let Some(helpful) = helpful_action_ids.as_deref_mut()
                {
                    helpful.push(OperatorIndex::from_usize(task_idx));
                }
                plan_cost += self.op_cost[op_id.index()];
                // Synthetic ops pull their parent in for cost accounting.
                if let Some(parent) = self.op_parent[op_id.index()]
                    && !scratch.in_plan[parent.index()]
                {
                    scratch.in_plan[parent.index()] = true;
                    if (0..=1).contains(&scratch.op_first_layer[parent.index()])
                        && let Some(task_idx) = self.op_task_idx[parent.index()]
                        && let Some(helpful) = helpful_action_ids.as_deref_mut()
                    {
                        helpful.push(OperatorIndex::from_usize(task_idx));
                    }
                    plan_cost += self.op_cost[parent.index()];
                }
                for &pre_fid in &self.op_preconditions[op_id.index()] {
                    if scratch.seen[pre_fid] {
                        continue;
                    }
                    let pre_layer = scratch.fact_first_layer[pre_fid];
                    if pre_layer <= 0 {
                        scratch.seen[pre_fid] = true;
                        continue;
                    }
                    debug_assert!(pre_layer < layer);
                    scratch.seen[pre_fid] = true;
                    if (pre_layer as usize) < scratch.goals_at_layer.len() {
                        scratch.goals_at_layer[pre_layer as usize].push(pre_fid);
                    }
                }
            }
        }
        plan_cost
    }
}

impl FfHeuristic {
    /// The cost of a relaxed plan for the state, infinite when the delete
    /// relaxation reaches no goal; with `helpful`, appending the plan's operators
    /// applicable in the state.
    fn relaxed_plan_cost(
        &mut self,
        eval_state: &EvaluationState<'_, '_>,
        helpful: Option<&mut Vec<OperatorIndex>>,
    ) -> Result<Estimate, EvaluationError> {
        if eval_state.is_goal() {
            return Ok(Estimate::new(0.0));
        }
        let mut scratch = std::mem::take(&mut self.scratch);
        scratch.reset();
        let goal_layer = self.build_rpg(eval_state, &mut scratch);
        let cost = match goal_layer {
            Err(error) => Err(error),
            // The delete relaxation reaches no goal: a dead end.
            Ok(i32::MAX) => Ok(Estimate::DEAD_END),
            Ok(0) => Ok(Estimate::new(0.0)),
            Ok(_) => Ok(Estimate::new(
                self.extract_relaxed_plan(&mut scratch, helpful),
            )),
        };
        self.scratch = scratch;
        cost
    }
}

impl Heuristic for FfHeuristic {
    fn evaluate(
        &mut self,
        eval_state: &EvaluationState<'_, '_>,
    ) -> Result<Estimate, EvaluationError> {
        self.relaxed_plan_cost(eval_state, None)
    }

    fn evaluate_with_preferred(
        &mut self,
        eval_state: &EvaluationState<'_, '_>,
        preferred: &mut Vec<OperatorIndex>,
    ) -> Result<Estimate, EvaluationError> {
        self.relaxed_plan_cost(eval_state, Some(preferred))
    }

    fn name(&self) -> &str {
        "ff"
    }
}

/// Registry feature for FF.
pub struct FfFeature;
impl planforge_search::heuristic_factory::HeuristicFeature for FfFeature {
    const NAME: &'static str = "ff";
    type Config = planforge_search::heuristic_factory::EmptyConfig;
    fn build(
        _: &Self::Config,
        context: &planforge_search::heuristic_factory::BuildContext,
    ) -> Result<Box<dyn Heuristic>, planforge_search::heuristic_factory::HeuristicBuildError> {
        let heuristic = FfHeuristic::new(&context.task)
            .map_err(|error| format!("failed to construct ff heuristic: {error}"))?;
        Ok(Box::new(heuristic))
    }
}
