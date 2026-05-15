//! Fast-Forward (h_FF) heuristic with monotonic numeric relaxation.
//!
//! Classical Hoffmann/Nebel relaxed-plan heuristic, extended with Metric-FF
//! style monotonic tracking of numeric variables. For each state:
//!
//!  1. Build a forward relaxed planning graph (RPG). Each layer adds
//!     propositional facts achievable through some operator's effects and
//!     newly-true comparison-axiom facts achievable through the per-
//!     numeric-var max/min envelope.
//!  2. Stop when every goal fact (including comparison-axiom-TRUE goal
//!     facts) is in the RPG or no progress is made.
//!  3. Backward-chain from the goal facts to pick supporters per layer;
//!     `h_FF(s)` is the total cost of the operators in the extracted
//!     relaxed plan.
//!
//! Numeric relaxation model: every numeric variable tracks a pair
//! `(max_reachable, min_reachable)` initialized from the current state.
//! Operator assignment effects update them monotonically:
//!
//!  * `Plus(a, rhs)`: relax `a` upward by `max[rhs]` if positive, downward
//!     by `-min[rhs]` if negative.
//!  * `Minus(a, rhs)`: symmetric.
//!  * `Assign(a, rhs)`: `max[a] := max(max[a], max[rhs])` and
//!     `min[a] := min(min[a], min[rhs])`.
//!  * `Times` / `Divide`: dropped — bounding these monotonically requires
//!     sign-aware case analysis the first pass doesn't do; treating them
//!     as no-ops is sound (relaxation only ever drops constraints, never
//!     adds them) but leaves them off the heuristic's radar.
//!
//! Comparison axioms (e.g. `(>= x v)`) become *available* as soon as the
//! relaxed numeric envelope makes them satisfiable: `(>= x v)` is true
//! when `max[x] >= min[v]` etc. Once available the comparison-axiom
//! TRUE fact behaves like any other RPG fact. Its achievers are the
//! operators whose assignment effects touch the LHS or RHS numeric
//! variable, which is what relaxed-plan extraction picks from.
//!
//! Reference: Hoffmann & Nebel, *The FF Planning System*, JAIR 2001;
//! Hoffmann, *The Metric-FF Planning System*, JAIR 2003.

use std::cell::RefCell;
use std::collections::{HashSet, VecDeque};

use crate::numeric::evaluation::evaluator::{EvaluationError, EvaluationState};
use crate::numeric::evaluation::heuristic::Heuristic;
use planforge_sas::numeric::axioms::ComparisonOperator;
use planforge_sas::numeric::numeric_task::{
    AbstractNumericTask, AssignmentOperation, ExplicitFact, NumericType,
    metric_operator_cost_from_initial_values,
};

type FactId = usize;
type OpId = usize;
type NumVarId = usize;

const COMPARISON_TRUE_VALUE: usize = 0;

/// Monotonic-relaxation entry for one numeric variable.
#[derive(Debug, Clone, Copy)]
struct NumericRange {
    max: f64,
    min: f64,
}

impl NumericRange {
    const fn singleton(v: f64) -> Self {
        Self { max: v, min: v }
    }

    fn join(&mut self, other: NumericRange) -> bool {
        let new_max = if other.max > self.max { other.max } else { self.max };
        let new_min = if other.min < self.min { other.min } else { self.min };
        let changed = new_max > self.max + f64::EPSILON || new_min < self.min - f64::EPSILON;
        self.max = new_max;
        self.min = new_min;
        changed
    }
}

#[derive(Debug, Clone)]
struct AssignmentEffectDesc {
    affected_var: NumVarId,
    operation: AssignmentOperation,
    rhs_var: NumVarId,
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

/// Per-state RPG buffers reused across `compute_heuristic` calls to avoid
/// re-allocating on every search node.
struct ScratchBuffers {
    fact_first_layer: Vec<i32>,
    op_remaining_preconditions: Vec<i32>,
    op_first_layer: Vec<i32>,
    queue: VecDeque<FactId>,
    goals_at_layer: Vec<Vec<FactId>>,
    seen: Vec<bool>,
    in_plan: Vec<bool>,
    numeric: Vec<NumericRange>,
    /// Per-comparison-axiom: layer at which it first became satisfiable
    /// (i32::MAX if never). Tracked so achiever lookup is layer-aware.
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
        for v in &mut self.axiom_first_layer {
            *v = -1;
        }
    }
}

pub struct FfHeuristic<'task> {
    /// Phantom borrow of the task — `FfHeuristic` is constructed from a
    /// `&'task dyn AbstractNumericTask` and all field data is derived from
    /// it, so the heuristic mustn't outlive that borrow.
    _task: std::marker::PhantomData<&'task ()>,
    /// Per-operator propositional preconditions (numeric-axiom-var
    /// preconditions are encoded as comparison-axiom-TRUE facts instead).
    op_preconditions: Vec<Vec<FactId>>,
    /// Per-operator propositional add-effects (numeric-axiom-var effects
    /// don't appear here — they're driven by the axioms).
    op_effects: Vec<Vec<FactId>>,
    /// Per-operator monotonic numeric effects (Plus / Minus / Assign with
    /// constant or live RHS).
    op_numeric_effects: Vec<Vec<AssignmentEffectDesc>>,
    op_cost: Vec<f64>,
    /// All goal facts (propositional and comparison-axiom-TRUE).
    goal_facts: Vec<FactId>,
    /// For each fact, list of operators that achieve it. For propositional
    /// facts: ops whose add-effects include it. For comparison-axiom-TRUE
    /// facts: ops whose numeric effects touch the LHS or RHS numeric
    /// variable.
    achievers: Vec<Vec<OpId>>,
    /// For each fact id, the operators that have it as a precondition.
    consumers: Vec<Vec<OpId>>,
    fact_var_value: Vec<(usize, usize)>,
    /// Numeric variables initial values for `Constant` types (used by
    /// numeric init so we don't need a state registry to read constants).
    constant_numeric: Vec<Option<f64>>,
    /// Comparison axioms in declaration order; indexed by `ComparisonAxiomDesc`.
    comparison_axioms: Vec<ComparisonAxiomDesc>,
    /// For each numeric var: indices into `comparison_axioms` for axioms
    /// that mention this var (LHS or RHS). Lets us re-evaluate only the
    /// affected axioms after a numeric effect.
    axioms_touching_var: Vec<Vec<usize>>,
    /// FactId -> Some(axiom index) if this fact is a comparison-axiom-TRUE
    /// fact; None for ordinary propositional facts.
    fact_to_axiom: Vec<Option<usize>>,
    num_facts: usize,
    num_numeric: usize,
    scratch: RefCell<ScratchBuffers>,
}

impl<'task> FfHeuristic<'task> {
    pub fn new(task: &'task dyn AbstractNumericTask) -> Result<Self, String> {
        // 1. Identify numeric-axiom variables (driven by comparison or
        //    assignment axioms). These don't get propositional facts —
        //    except for the comparison-axiom TRUE value, which is what
        //    other operators precondition on.
        let mut numeric_axiom_vars: HashSet<usize> = HashSet::new();
        for axiom in task.comparison_axioms() {
            numeric_axiom_vars.insert(axiom.get_affected_var_id());
        }
        for axiom in task.assignment_axioms() {
            numeric_axiom_vars.insert(axiom.get_affected_var_id());
        }

        // 2. Enumerate facts:
        //    - For each non-numeric-axiom variable, one FactId per value.
        //    - For each comparison-axiom variable, exactly one FactId for
        //      its TRUE value (FALSE / UNKNOWN values are dropped — the
        //      relaxation is monotonic upward).
        let num_props = task.variables().len();
        let mut fact_id_table: Vec<Vec<Option<FactId>>> = (0..num_props)
            .map(|var_id| {
                let var = &task.variables()[var_id];
                vec![None; var.domain_size()]
            })
            .collect();
        let mut fact_var_value: Vec<(usize, usize)> = Vec::new();
        let mut fact_to_axiom: Vec<Option<usize>> = Vec::new();

        let mut comparison_axioms = Vec::with_capacity(task.comparison_axioms().len());
        for var_id in 0..num_props {
            if numeric_axiom_vars.contains(&var_id) {
                continue;
            }
            let range = task.variables()[var_id].domain_size();
            for value in 0..range {
                let fid = fact_var_value.len();
                fact_id_table[var_id][value] = Some(fid);
                fact_var_value.push((var_id, value));
                fact_to_axiom.push(None);
            }
        }
        // Now add one FactId per comparison-axiom-TRUE fact and register the
        // axiom descriptor.
        for (axiom_idx, axiom) in task.comparison_axioms().iter().enumerate() {
            let fid = fact_var_value.len();
            let affected = axiom.get_affected_var_id();
            // Reserve a row entry for this variable's TRUE value so
            // `map_fact` resolves it.
            let row = &mut fact_id_table[affected];
            if row.is_empty() {
                *row = vec![None; task.variables()[affected].domain_size()];
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
            // (We intentionally do not silence un-used `axiom_idx` — the
            // descriptor's storage order is the canonical axiom index.)
            let _ = axiom_idx;
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

        // 3. Index axioms by numeric-var membership so we don't have to
        //    re-scan the full axiom list after every numeric update.
        let num_numeric = task.numeric_variables().len();
        let mut axioms_touching_var: Vec<Vec<usize>> = vec![Vec::new(); num_numeric];
        for (idx, ax) in comparison_axioms.iter().enumerate() {
            axioms_touching_var[ax.left_var].push(idx);
            if ax.right_var != ax.left_var {
                axioms_touching_var[ax.right_var].push(idx);
            }
        }

        // 4. Pre-compute per-operator preconditions / effects /
        //    numeric-effect descriptors.
        let operators = task.get_operators();
        let mut op_preconditions = Vec::with_capacity(operators.len());
        let mut op_effects = Vec::with_capacity(operators.len());
        let mut op_numeric_effects = Vec::with_capacity(operators.len());
        let mut op_cost = Vec::with_capacity(operators.len());
        let mut achievers: Vec<Vec<OpId>> = vec![Vec::new(); num_facts];

        for (op_idx, op) in operators.iter().enumerate() {
            let preconditions: Vec<FactId> =
                op.preconditions().iter().filter_map(map_fact).collect();

            let mut effects: Vec<FactId> = Vec::new();
            for eff in op.effects() {
                if !eff.conditions().is_empty() {
                    continue; // conditional effects: not modeled
                }
                if let Some(fid) = map_fact(&ExplicitFact::new(eff.var_id(), eff.value())) {
                    effects.push(fid);
                    achievers[fid].push(op_idx);
                }
            }

            // Numeric effects: keep only operations we can monotonically
            // relax (Plus/Minus/Assign). Times and Divide require sign-
            // aware bounding we don't implement; dropping is sound.
            let mut numeric_effects = Vec::new();
            for assign in op.assignment_effects() {
                if !assign.conditions().is_empty() {
                    continue;
                }
                match assign.operation() {
                    AssignmentOperation::Plus
                    | AssignmentOperation::Minus
                    | AssignmentOperation::Assign => {
                        numeric_effects.push(AssignmentEffectDesc {
                            affected_var: assign.affected_var_id(),
                            operation: assign.operation().clone(),
                            rhs_var: assign.var_id(),
                        });
                    }
                    AssignmentOperation::Times | AssignmentOperation::Divide => {}
                }
            }

            // Wire the operator as a potential achiever for any comparison
            // axiom whose LHS or RHS variable it modifies. This is how
            // comparison-axiom TRUE facts get supporters during relaxed-
            // plan extraction.
            for eff in &numeric_effects {
                for axiom_idx in &axioms_touching_var[eff.affected_var] {
                    let fid = comparison_axioms[*axiom_idx].true_fact;
                    if !achievers[fid].contains(&op_idx) {
                        achievers[fid].push(op_idx);
                    }
                }
            }

            let cost = metric_operator_cost_from_initial_values(task, op).max(0.0);
            op_preconditions.push(preconditions);
            op_effects.push(effects);
            op_numeric_effects.push(numeric_effects);
            op_cost.push(cost);
        }

        // 5. Goal facts.
        let goal_facts: Vec<FactId> = (0..task.get_num_goals())
            .filter_map(|i| {
                let goal = task.get_goal_fact(i);
                map_fact(goal)
            })
            .collect();

        // 6. Constant numeric values (for relaxation init we read the live
        //    state, but constants don't move — capture them once).
        let initial_numeric = task.get_initial_numeric_state_values();
        let constant_numeric: Vec<Option<f64>> = task
            .numeric_variables()
            .iter()
            .enumerate()
            .map(|(idx, var)| match var.get_type() {
                NumericType::Constant => initial_numeric.get(idx).copied(),
                _ => None,
            })
            .collect();
        drop(initial_numeric);

        // 7. Per-fact "operators that need it as a precondition" — built
        //    once so the BFS doesn't re-scan every layer.
        let mut consumers: Vec<Vec<OpId>> = vec![Vec::new(); num_facts];
        for (op_idx, prec) in op_preconditions.iter().enumerate() {
            for &fid in prec {
                consumers[fid].push(op_idx);
            }
        }

        Ok(Self {
            _task: std::marker::PhantomData,
            op_preconditions,
            op_effects,
            op_numeric_effects,
            op_cost,
            goal_facts,
            achievers,
            consumers,
            fact_var_value,
            constant_numeric,
            comparison_axioms,
            axioms_touching_var,
            fact_to_axiom,
            num_facts,
            num_numeric,
            scratch: RefCell::new(ScratchBuffers::new(
                num_facts,
                operators.len(),
                task.numeric_variables().len(),
                task.comparison_axioms().len(),
            )),
        })
    }

    fn state_holds_fact(
        &self,
        eval_state: &EvaluationState<'_, '_>,
        fid: FactId,
    ) -> bool {
        let (var, value) = self.fact_var_value[fid];
        let fact = ExplicitFact::new(var, value);
        let Some(registry) = eval_state.state_registry() else {
            return false;
        };
        fact.is_hold(eval_state.state(), registry)
    }

    fn initial_numeric_state(
        &self,
        eval_state: &EvaluationState<'_, '_>,
    ) -> Vec<NumericRange> {
        let mut out: Vec<NumericRange> = Vec::with_capacity(self.num_numeric);
        if let Some(registry) = eval_state.state_registry() {
            let mut buffer: Vec<f64> = Vec::new();
            if registry
                .fill_numeric_vars(eval_state.state(), &mut buffer)
                .is_ok()
            {
                for v in buffer {
                    out.push(NumericRange::singleton(v));
                }
            }
        }
        // Pad / fallback with declared constants for any missing entries.
        while out.len() < self.num_numeric {
            let idx = out.len();
            let v = self
                .constant_numeric
                .get(idx)
                .copied()
                .flatten()
                .unwrap_or(0.0);
            out.push(NumericRange::singleton(v));
        }
        out.truncate(self.num_numeric);
        out
    }

    fn evaluate_axiom(&self, axiom: &ComparisonAxiomDesc, numeric: &[NumericRange]) -> bool {
        if axiom.left_var >= numeric.len() || axiom.right_var >= numeric.len() {
            return false;
        }
        let l = numeric[axiom.left_var];
        let r = numeric[axiom.right_var];
        match axiom.op {
            ComparisonOperator::LessThan => l.min < r.max,
            ComparisonOperator::LessThanOrEqual => l.min <= r.max,
            ComparisonOperator::Equal => l.min <= r.max && l.max >= r.min,
            ComparisonOperator::GreaterThanOrEqual => l.max >= r.min,
            ComparisonOperator::GreaterThan => l.max > r.min,
            ComparisonOperator::UnEqual => {
                // Two ranges are unequal as soon as they don't coincide at
                // a single point. With singleton-init, until any numeric
                // effect fires, this matches the strict-equality test from
                // the initial state.
                l.min != l.max || r.min != r.max || l.min != r.min
            }
        }
    }

    fn apply_numeric_effect(
        &self,
        eff: &AssignmentEffectDesc,
        numeric: &mut [NumericRange],
    ) -> bool {
        if eff.rhs_var >= numeric.len() || eff.affected_var >= numeric.len() {
            return false;
        }
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
                    next.max = prev.max + rhs.max;
                }
                if rhs.min < 0.0 {
                    next.min = prev.min + rhs.min;
                }
                next
            }
            AssignmentOperation::Minus => {
                let mut next = prev;
                if rhs.min < 0.0 {
                    next.max = prev.max - rhs.min;
                }
                if rhs.max > 0.0 {
                    next.min = prev.min - rhs.max;
                }
                next
            }
            // Filtered out at construction; unreachable.
            AssignmentOperation::Times | AssignmentOperation::Divide => prev,
        };
        numeric[eff.affected_var].join(new)
    }

    /// Build the RPG. Returns the max layer at which a goal first appears,
    /// or `i32::MAX` if some goal is never reached.
    fn build_rpg(
        &self,
        eval_state: &EvaluationState<'_, '_>,
        scratch: &mut ScratchBuffers,
    ) -> i32 {
        scratch.numeric = self.initial_numeric_state(eval_state);

        // Layer 0 propositional facts.
        for fid in 0..self.num_facts {
            if self.fact_to_axiom[fid].is_some() {
                continue; // axiom facts evaluated below
            }
            if self.state_holds_fact(eval_state, fid) {
                scratch.fact_first_layer[fid] = 0;
                scratch.queue.push_back(fid);
            }
        }

        // Layer 0 comparison-axiom TRUE facts: any axiom satisfied under
        // the initial relaxed numeric envelope (which is just the live
        // state).
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
        // Empty-precondition operators fire immediately at layer 0.
        for (op_id, prec) in self.op_preconditions.iter().enumerate() {
            if prec.is_empty() {
                self.fire_operator(op_id, 0, scratch);
            }
        }
        if self.goal_satisfied(scratch) {
            return self.goal_max_layer(scratch);
        }

        // Main BFS loop: pop a newly-reached fact, decrement its consumer
        // operators' remaining counters, fire any operators that just hit
        // zero. Firing applies numeric effects, which can in turn unlock
        // comparison-axiom TRUE facts.
        while let Some(fid) = scratch.queue.pop_front() {
            let fact_layer = scratch.fact_first_layer[fid];
            for &op_id in &self.consumers[fid] {
                let remaining = &mut scratch.op_remaining_preconditions[op_id];
                if *remaining > 0 {
                    *remaining -= 1;
                    if *remaining == 0 {
                        self.fire_operator(op_id, fact_layer + 1, scratch);
                    }
                }
            }
            if self.goal_satisfied(scratch) {
                return self.goal_max_layer(scratch);
            }
        }

        if self.goal_satisfied(scratch) {
            self.goal_max_layer(scratch)
        } else {
            i32::MAX
        }
    }

    fn fire_operator(
        &self,
        op_id: OpId,
        layer: i32,
        scratch: &mut ScratchBuffers,
    ) {
        if scratch.op_first_layer[op_id] >= 0 {
            return;
        }
        scratch.op_first_layer[op_id] = layer;

        // Propositional add-effects.
        for &fid in &self.op_effects[op_id] {
            if scratch.fact_first_layer[fid] < 0 {
                scratch.fact_first_layer[fid] = layer;
                scratch.queue.push_back(fid);
            }
        }

        // Numeric effects. Apply, then re-evaluate any axiom that touches
        // the affected variable; newly-true axioms emit their TRUE fact at
        // this layer.
        let mut dirty_axioms: HashSet<usize> = HashSet::new();
        for eff in &self.op_numeric_effects[op_id] {
            if self.apply_numeric_effect(eff, &mut scratch.numeric) {
                for &ax in &self.axioms_touching_var[eff.affected_var] {
                    dirty_axioms.insert(ax);
                }
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

    /// Backward relaxed-plan extraction with greedy cheapest-supporter
    /// selection. For each goal fact at its first-reachable layer, walk
    /// layers high-to-low and pick a cheapest achiever per fact; add the
    /// chosen achiever's preconditions to lower layers as new sub-goals.
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
                    let cost = self.op_cost[op_id];
                    if cost < best_cost {
                        best_cost = cost;
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
            return Ok(0.0);
        }
        let mut scratch = self.scratch.borrow_mut();
        scratch.reset();
        let goal_layer = self.build_rpg(eval_state, &mut scratch);
        if goal_layer == i32::MAX {
            return Err(EvaluationError::DeadEnd { reliable: false });
        }
        if goal_layer == 0 {
            return Ok(0.0);
        }
        Ok(self.extract_relaxed_plan(&mut scratch))
    }

    fn heuristic_name(&self) -> String {
        "ff".to_string()
    }
}
