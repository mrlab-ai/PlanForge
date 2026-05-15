//! Fast-Forward (h_FF) heuristic for the grounded numeric task.
//!
//! This is the classical Hoffmann/Nebel relaxed-plan heuristic. For each
//! state it (i) builds the relaxed planning graph (RPG) by propagating
//! propositional facts forward, (ii) finds the earliest layer where every
//! goal fact appears, then (iii) backward-chains from the goal facts to
//! extract a relaxed plan — the set of operators "needed" to reach the goal
//! under the delete relaxation. `h_FF(s)` is the sum of those operators'
//! costs.
//!
//! Numeric handling (current pass): numeric / comparison-axiom preconditions
//! and effects are treated as trivially achievable / no-op. That is sound
//! (admissible) — the relaxation drops constraints rather than adding any —
//! but loses informativeness on numeric-heavy tasks. Adding monotonic
//! numeric relaxation (Metric-FF style) is a follow-up; that is a non-
//! trivial extension and out of scope for this first cut.
//!
//! Useful as a fast non-admissible (in general) guide for greedy best-
//! first search, especially on propositional-dominated tasks.
//!
//! Reference: Hoffmann & Nebel, *The FF Planning System*, JAIR 2001.

use std::collections::VecDeque;

use std::collections::HashSet;

use crate::numeric::evaluation::evaluator::{EvaluationError, EvaluationState};
use crate::numeric::evaluation::heuristic::Heuristic;
use planforge_sas::numeric::numeric_task::{
    AbstractNumericTask, ExplicitFact, metric_operator_cost_from_initial_values,
};

/// Per-state RPG buffers reused across `compute_heuristic` calls to avoid
/// re-allocating on every search-node evaluation.
struct ScratchBuffers {
    fact_first_layer: Vec<i32>,
    op_remaining_preconditions: Vec<i32>,
    op_first_layer: Vec<i32>,
    queue: VecDeque<usize>,
    goals_at_layer: Vec<Vec<usize>>,
    seen: Vec<bool>,
    in_plan: Vec<bool>,
}

impl ScratchBuffers {
    fn new(num_facts: usize, num_ops: usize) -> Self {
        Self {
            fact_first_layer: vec![-1; num_facts],
            op_remaining_preconditions: Vec::with_capacity(num_ops),
            op_first_layer: vec![-1; num_ops],
            queue: VecDeque::new(),
            goals_at_layer: Vec::new(),
            seen: vec![false; num_facts],
            in_plan: vec![false; num_ops],
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
    }
}

pub struct FfHeuristic<'task> {
    task: &'task dyn AbstractNumericTask,
    /// Propositional preconditions per operator (filtered: numeric-axiom var
    /// preconditions dropped — see module docs).
    operator_preconditions: Vec<Vec<FactId>>,
    /// Propositional effects per operator (filtered same as above).
    operator_effects: Vec<Vec<FactId>>,
    operator_cost: Vec<f64>,
    /// Propositional goals (numeric-axiom-var goals dropped).
    goal_facts: Vec<FactId>,
    /// For each fact id, the operators that achieve it.
    achievers: Vec<Vec<usize>>,
    /// Inverse of the `(var, value) -> FactId` table; lets us check whether
    /// the initial state holds a given fact.
    fact_var_value: Vec<(usize, usize)>,
    num_facts: usize,
    scratch: std::cell::RefCell<ScratchBuffers>,
}

type FactId = usize;

impl<'task> FfHeuristic<'task> {
    pub fn new(task: &'task dyn AbstractNumericTask) -> Result<Self, String> {
        // Identify variables driven by comparison or assignment axioms.
        // These are excluded from the propositional RPG — their evolution
        // is governed by numeric state, which the relaxation here doesn't
        // model. (Future numeric extension would add monotonic min/max
        // tracking for each numeric variable.)
        let mut numeric_axiom_vars: HashSet<usize> = HashSet::new();
        for axiom in task.comparison_axioms() {
            numeric_axiom_vars.insert(axiom.get_affected_var_id());
        }
        for axiom in task.assignment_axioms() {
            numeric_axiom_vars.insert(axiom.get_affected_var_id());
        }
        let is_numeric_axiom_var = |var: usize| numeric_axiom_vars.contains(&var);

        let num_props = task.variables().len();
        let mut fact_id_table: Vec<Vec<Option<FactId>>> = (0..num_props)
            .map(|var_id| {
                let var = &task.variables()[var_id];
                vec![None; var.domain_size()]
            })
            .collect();
        let mut fact_var_value: Vec<(usize, usize)> = Vec::new();
        for var_id in 0..num_props {
            if is_numeric_axiom_var(var_id) {
                continue;
            }
            let range = task.variables()[var_id].domain_size();
            for value in 0..range {
                let fid = fact_var_value.len();
                fact_id_table[var_id][value] = Some(fid);
                fact_var_value.push((var_id, value));
            }
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

        let operators = task.get_operators();
        let mut operator_preconditions = Vec::with_capacity(operators.len());
        let mut operator_effects = Vec::with_capacity(operators.len());
        let mut operator_cost = Vec::with_capacity(operators.len());
        let mut achievers: Vec<Vec<usize>> = vec![Vec::new(); num_facts];

        for (op_idx, op) in operators.iter().enumerate() {
            let preconditions: Vec<FactId> = op
                .preconditions()
                .iter()
                .filter_map(map_fact)
                .collect();
            // Only collect propositional, unconditional effects. Conditional
            // effects could be supported by treating them as additional
            // operator clones; for the first cut we drop them. Dropping is
            // sound — it removes potential achievers, never adds spurious
            // ones — and effects on numeric-axiom vars are not represented
            // in the propositional RPG anyway.
            let mut effects: Vec<FactId> = Vec::new();
            for eff in op.effects() {
                if eff.conditions().is_empty()
                    && let Some(fid) = map_fact(&ExplicitFact::new(eff.var_id(), eff.value()))
                {
                    effects.push(fid);
                    achievers[fid].push(op_idx);
                }
            }
            // Operator cost — use the metric-aware cost when a metric is
            // declared, otherwise the SAS-declared integer cost.
            let cost = metric_operator_cost_from_initial_values(task, op);
            operator_preconditions.push(preconditions);
            operator_effects.push(effects);
            operator_cost.push(cost.max(0.0));
        }

        let goal_facts: Vec<FactId> = (0..task.get_num_goals())
            .filter_map(|i| {
                let goal = task.get_goal_fact(i);
                map_fact(goal)
            })
            .collect();

        Ok(Self {
            task,
            operator_preconditions,
            operator_effects,
            operator_cost,
            goal_facts,
            achievers,
            fact_var_value,
            num_facts,
            scratch: std::cell::RefCell::new(ScratchBuffers::new(num_facts, operators.len())),
        })
    }

    /// Test whether `state` holds the (var, value) pair identified by `fid`.
    /// Returns `false` if no state registry is available (which would be a
    /// programmer error in the search driver — the FF heuristic is only
    /// useful when invoked via the registry-backed search path).
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

    /// Build the RPG forward pass, returning the layer at which the relaxed
    /// goal is satisfied. `i32::MAX` if the goal isn't reachable.
    fn build_rpg(
        &self,
        eval_state: &EvaluationState<'_, '_>,
        scratch: &mut ScratchBuffers,
    ) -> i32 {
        // Initial layer: facts that hold in the state.
        for fid in 0..self.num_facts {
            if self.state_holds_fact(eval_state, fid) {
                scratch.fact_first_layer[fid] = 0;
                scratch.queue.push_back(fid);
            }
        }

        // Operators with empty preconditions become applicable in layer 0
        // too — handle them explicitly so they don't get stuck on the
        // "decrement remaining" path.
        scratch
            .op_remaining_preconditions
            .resize(self.operator_preconditions.len(), 0);
        for (op_id, prec) in self.operator_preconditions.iter().enumerate() {
            scratch.op_remaining_preconditions[op_id] = prec.len() as i32;
        }
        for (op_id, prec) in self.operator_preconditions.iter().enumerate() {
            if prec.is_empty() {
                self.apply_relaxed_operator(op_id, 0, scratch);
            }
        }

        // BFS layer propagation. Each fact, when first reached, "fires" the
        // operators it preconditions. When an operator's last precondition
        // is fired, the operator becomes applicable and its effects are
        // emitted at the next layer.
        //
        // Index per-fact `consumers` so the BFS doesn't scan all operators
        // every layer.
        let consumers = self.fact_consumers();
        while let Some(fid) = scratch.queue.pop_front() {
            let fact_layer = scratch.fact_first_layer[fid];
            for &op_id in &consumers[fid] {
                let remaining = &mut scratch.op_remaining_preconditions[op_id];
                if *remaining > 0 {
                    *remaining -= 1;
                    if *remaining == 0 {
                        self.apply_relaxed_operator(op_id, fact_layer + 1, scratch);
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

    /// Build the fact → operators-that-need-it index lazily on each call.
    /// This is O(operators × avg-precondition-len) which is acceptable —
    /// the cost is dwarfed by the BFS itself.
    fn fact_consumers(&self) -> Vec<Vec<usize>> {
        let mut consumers: Vec<Vec<usize>> = vec![Vec::new(); self.num_facts];
        for (op_id, prec) in self.operator_preconditions.iter().enumerate() {
            for &fid in prec {
                consumers[fid].push(op_id);
            }
        }
        consumers
    }

    fn apply_relaxed_operator(
        &self,
        op_id: usize,
        layer: i32,
        scratch: &mut ScratchBuffers,
    ) {
        if scratch.op_first_layer[op_id] >= 0 {
            return;
        }
        scratch.op_first_layer[op_id] = layer;
        for &fid in &self.operator_effects[op_id] {
            if scratch.fact_first_layer[fid] < 0 {
                scratch.fact_first_layer[fid] = layer;
                scratch.queue.push_back(fid);
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

    /// Backward relaxed-plan extraction. For each goal fact, queue it at its
    /// first-reachable layer. Walking layers high-to-low, for each queued
    /// fact pick a cheapest achiever that fired strictly before it; add the
    /// achiever's preconditions to lower layers. Operators added this way
    /// constitute the relaxed plan.
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
            if layer < 0 {
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
                // Find a cheapest achiever that fired at a layer strictly
                // less than `layer`. (Operators fire at `prev_layer + 1`, so
                // an op firing at layer `layer-1` produces a fact at layer
                // `layer-1`; we need an op whose own layer is `< layer`,
                // i.e. `<= layer - 1`.)
                let target_op_layer = layer - 1;
                let mut best_op: Option<usize> = None;
                let mut best_cost = f64::INFINITY;
                for &op_id in &self.achievers[fid] {
                    let op_layer = scratch.op_first_layer[op_id];
                    if op_layer < 0 || op_layer > target_op_layer {
                        continue;
                    }
                    let cost = self.operator_cost[op_id];
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
                plan_cost += self.operator_cost[op_id];
                // Add the op's preconditions at their respective layers as
                // new sub-goals.
                for &pre_fid in &self.operator_preconditions[op_id] {
                    if scratch.seen[pre_fid] {
                        continue;
                    }
                    let pre_layer = scratch.fact_first_layer[pre_fid];
                    if pre_layer <= 0 {
                        // Initial-layer facts are free.
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

    pub fn name(&self) -> &str {
        "ff"
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
