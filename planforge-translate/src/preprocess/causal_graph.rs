//! The causal graph of a translated task, and the ordering it induces.
//!
//! An edge runs from every variable an operator or axiom reads to every
//! variable it writes, weighted by how often that dependency occurs. The
//! variables are then ordered by a topological sort of the graph's strongly
//! connected components, and the ones nothing the task asks for depends on are
//! dropped.

use std::collections::BTreeMap;

use tracing::{debug, info};

use super::max_dag::MaxDag;
use super::scc::Scc;
use super::{
    NO_LAYER, NO_LEVEL, NumType, NumericVarState, PreprocessedTask, ReorderedTask, VarState,
};
use crate::sas_tasks::{SASAxiom, SASCompareAxiom, SASNumericAxiom, SASOperator, SasFact};

/// A node of the causal graph: the task numbers its propositional and its
/// numeric variables independently, so a node needs both.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum EitherVar {
    ExplicitVariable(usize),
    NumericVariable(usize),
}

pub type WeightedSuccessors = BTreeMap<EitherVar, u64>;
pub type WeightedGraph = BTreeMap<EitherVar, WeightedSuccessors>;
pub type Predecessors = BTreeMap<EitherVar, u64>;
pub type PredecessorGraph = BTreeMap<EitherVar, Predecessors>;

pub type Partition = Vec<Vec<EitherVar>>;

pub struct CausalGraph {
    task: PreprocessedTask,
    predecessor_graph: PredecessorGraph,

    /// Every variable of the task, in causal-graph order.
    ordering: Vec<EitherVar>,
    /// The surviving variables of each kind, in causal-graph order.
    prop_order: Vec<usize>,
    numeric_order: Vec<usize>,
}

impl CausalGraph {
    pub fn new(task: PreprocessedTask) -> Self {
        let sas = &task.sas;
        let mut weighted_graph: WeightedGraph = BTreeMap::new();
        for var in 0..task.vars.len() {
            weighted_graph.insert(EitherVar::ExplicitVariable(var), BTreeMap::new());
        }
        for var in 0..task.numeric_vars.len() {
            weighted_graph.insert(EitherVar::NumericVariable(var), BTreeMap::new());
        }

        let mut predecessor_graph: PredecessorGraph = BTreeMap::new();
        weigh_operators(&sas.operators, &mut weighted_graph, &mut predecessor_graph);
        weigh_axioms(&sas.axioms, &mut weighted_graph, &mut predecessor_graph);
        weigh_comparison_axioms(
            &sas.comp_axioms,
            &mut weighted_graph,
            &mut predecessor_graph,
        );
        weigh_numeric_axioms(
            &sas.numeric_axioms,
            &mut weighted_graph,
            &mut predecessor_graph,
        );

        let sccs = strongly_connected_components(
            task.vars.len(),
            task.numeric_vars.len(),
            &weighted_graph,
        );
        let ordering = topological_pseudo_sort(&task, &weighted_graph, &sccs);

        info!(
            "The causal graph is {}acyclic.",
            if sccs.len() == task.vars.len() + task.numeric_vars.len() {
                ""
            } else {
                "not "
            }
        );

        let mut cg = Self {
            task,
            predecessor_graph,
            ordering,
            prop_order: Vec::new(),
            numeric_order: Vec::new(),
        };

        cg.place_necessary_variables();

        cg
    }

    /// Marks the variables the task cannot be solved without -- the ones a
    /// goal, the global constraint or the metric depends on, transitively --
    /// and gives each of them its position in the reordered task.
    fn place_necessary_variables(&mut self) {
        let PreprocessedTask {
            sas,
            metric,
            vars,
            numeric_vars,
        } = &mut self.task;

        for &(var, _) in &sas.goal.pairs {
            if !vars[var].is_necessary() {
                debug!("var{var} is directly necessary (goal).");
                vars[var].mark_necessary();
                mark_predecessors_necessary(
                    &self.predecessor_graph,
                    EitherVar::ExplicitVariable(var),
                    vars,
                    numeric_vars,
                );
            }
        }

        let (constraint_var, _) = sas.global_constraint;
        if !vars[constraint_var].is_necessary() {
            debug!("var{constraint_var} is directly necessary (global constraint).");
            vars[constraint_var].mark_necessary();
            mark_predecessors_necessary(
                &self.predecessor_graph,
                EitherVar::ExplicitVariable(constraint_var),
                vars,
                numeric_vars,
            );
        }

        mark_instrumentation_necessary(&sas.numeric_axioms, numeric_vars, metric.index);
        for op in &sas.operators {
            for &(var, _, operand, _) in &op.assign_effects {
                if numeric_vars[var].ntype() == NumType::Instrumentation {
                    assert!(numeric_vars[var].is_necessary());
                    mark_instrumentation_necessary(&sas.numeric_axioms, numeric_vars, operand);
                }
            }
        }

        assert!(self.prop_order.is_empty());
        assert!(self.numeric_order.is_empty());
        assert_eq!(self.ordering.len(), vars.len() + numeric_vars.len());
        for cg_var in &self.ordering {
            match *cg_var {
                EitherVar::ExplicitVariable(var) => {
                    if vars[var].is_necessary() {
                        vars[var].set_level(self.prop_order.len() as i32);
                        self.prop_order.push(var);
                    }
                }
                EitherVar::NumericVariable(var) => {
                    if numeric_vars[var].is_necessary() {
                        numeric_vars[var].set_level(self.numeric_order.len() as i32);
                        self.numeric_order.push(var);
                    }
                }
            }
        }
        info!(
            "{} variables of {} necessary",
            self.prop_order.len(),
            vars.len()
        );
        info!(
            "{} numeric variables of {} necessary",
            self.numeric_order.len(),
            numeric_vars.len()
        );
    }

    /// Drops the effects, mutex groups and axioms that speak about a pruned
    /// variable, and the operators left without an effect.
    fn strip_pruned_variables(&mut self) {
        let PreprocessedTask {
            sas,
            vars,
            numeric_vars,
            ..
        } = &mut self.task;

        let mutexes_before = sas.mutexes.len();
        for mutex in &mut sas.mutexes {
            mutex
                .facts
                .retain(|&(var, _)| vars[var].level() != NO_LEVEL);
        }
        // A group that has shrunk to facts on a single variable states nothing:
        // two values of one variable are mutually exclusive by construction.
        sas.mutexes
            .retain(|mutex| mutex.facts.windows(2).any(|pair| pair[0].0 != pair[1].0));
        info!(
            "{} of {} mutex groups necessary.",
            sas.mutexes.len(),
            mutexes_before
        );

        let operators_before = sas.operators.len();
        for op in &mut sas.operators {
            op.pre_post
                .retain(|&(var, ..)| vars[var].level() != NO_LEVEL);
            op.assign_effects
                .retain(|&(var, ..)| numeric_vars[var].level() != NO_LEVEL);
        }
        sas.operators
            .retain(|op| !is_redundant_operator(op, numeric_vars));
        info!(
            "{} of {} operators necessary.",
            sas.operators.len(),
            operators_before
        );

        let axioms_before = sas.axioms.len();
        sas.axioms
            .retain(|axiom| vars[axiom.effect.0].level() != NO_LEVEL);
        info!(
            "{} of {} axiom rules necessary.",
            sas.axioms.len(),
            axioms_before
        );

        let comp_axioms_before = sas.comp_axioms.len();
        sas.comp_axioms.retain(|axiom| {
            vars[axiom.effect].level() != NO_LEVEL
                && axiom
                    .parts
                    .iter()
                    .all(|&part| numeric_vars[part].level() != NO_LEVEL)
        });
        info!(
            "{} of {} axiom functional comparison rules necessary.",
            sas.comp_axioms.len(),
            comp_axioms_before
        );

        let numeric_axioms_before = sas.numeric_axioms.len();
        sas.numeric_axioms.retain(|axiom| {
            numeric_vars[axiom.effect].level() != NO_LEVEL
                && axiom
                    .parts
                    .iter()
                    .all(|&part| numeric_vars[part].level() != NO_LEVEL)
        });
        info!(
            "{} of {} axiom functional assignment rules necessary.",
            sas.numeric_axioms.len(),
            numeric_axioms_before
        );
    }

    /// Pruning a numeric variable can empty the topmost axiom layers, and the
    /// search reads the layers as a contiguous range, so the propositional
    /// layers move down by however many numeric layers went away.
    fn close_axiom_layer_gap(&mut self) {
        let PreprocessedTask {
            sas, numeric_vars, ..
        } = &mut self.task;

        let mut top_layer_before = NO_LAYER;
        let mut top_layer_after = NO_LAYER;
        for (var, state) in numeric_vars.iter().enumerate() {
            let layer = sas.numeric_variables.axiom_layers[var];
            top_layer_before = top_layer_before.max(layer);
            if state.is_necessary() {
                top_layer_after = top_layer_after.max(layer);
            }
        }
        if top_layer_before == top_layer_after {
            return;
        }

        debug!("numeric axiom layers end at {top_layer_after}, not {top_layer_before}");
        let decrement = top_layer_before - top_layer_after;
        for layer in &mut sas.variables.axiom_layers {
            if *layer != NO_LAYER {
                *layer -= decrement;
                assert!(
                    *layer > top_layer_after,
                    "a propositional axiom layer sank into the numeric ones"
                );
            }
        }
    }

    /// Drops what the pruning made unreachable and answers with the task under
    /// its new variable numbering.
    pub fn finalize(mut self) -> ReorderedTask {
        self.strip_pruned_variables();
        self.close_axiom_layer_gap();

        let metric_level = self.task.numeric_vars[self.task.metric.index].level();
        self.task.metric.index = usize::try_from(metric_level)
            .expect("the metric variable is necessary, so the reordering gave it a level");

        ReorderedTask {
            task: self.task,
            prop_order: self.prop_order,
            numeric_order: self.numeric_order,
        }
    }
}

/// Records that `target` depends on `source`, unless a variable depends on
/// itself, which says nothing about the order the variables have to come in.
fn add_edge(
    weighted_graph: &mut WeightedGraph,
    predecessor_graph: &mut PredecessorGraph,
    source: EitherVar,
    target: EitherVar,
) {
    let successors = weighted_graph.entry(source).or_default();
    let predecessors = predecessor_graph.entry(target).or_default();
    if source != target {
        *successors.entry(target).or_insert(0) += 1;
        *predecessors.entry(source).or_insert(0) += 1;
    }
}

/// An operator's effect on a variable depends on everything the operator reads:
/// its preconditions, and the condition of that effect.
fn weigh_operators(
    operators: &[SASOperator],
    weighted_graph: &mut WeightedGraph,
    predecessor_graph: &mut PredecessorGraph,
) {
    let mut source_vars: Vec<EitherVar> = Vec::new();
    for op in operators {
        source_vars.clear();
        for &(var, _) in &op.prevail {
            source_vars.push(EitherVar::ExplicitVariable(var));
        }
        for &(var, pre, _, _) in &op.pre_post {
            if pre != -1 {
                source_vars.push(EitherVar::ExplicitVariable(var));
            }
        }
        let precondition_count = source_vars.len();

        for (var, _, _, conditions) in &op.pre_post {
            let target = EitherVar::ExplicitVariable(*var);
            extend_with_facts(&mut source_vars, conditions);
            for &source in &source_vars {
                add_edge(weighted_graph, predecessor_graph, source, target);
            }
            source_vars.truncate(precondition_count);
        }

        for (var, _, operand, conditions) in &op.assign_effects {
            let target = EitherVar::NumericVariable(*var);
            extend_with_facts(&mut source_vars, conditions);
            source_vars.push(EitherVar::NumericVariable(*operand));
            for &source in &source_vars {
                add_edge(weighted_graph, predecessor_graph, source, target);
            }
            source_vars.truncate(precondition_count);
        }
    }
}

fn extend_with_facts(source_vars: &mut Vec<EitherVar>, facts: &[SasFact]) {
    source_vars.extend(
        facts
            .iter()
            .map(|&(var, _)| EitherVar::ExplicitVariable(var)),
    );
}

fn weigh_axioms(
    axioms: &[SASAxiom],
    weighted_graph: &mut WeightedGraph,
    predecessor_graph: &mut PredecessorGraph,
) {
    for axiom in axioms {
        let target = EitherVar::ExplicitVariable(axiom.effect.0);
        for &(var, _) in &axiom.condition {
            let source = EitherVar::ExplicitVariable(var);
            add_edge(weighted_graph, predecessor_graph, source, target);
        }
    }
}

/// A comparison variable depends on both sides of the comparison. The two are
/// numeric and the variable is propositional, so neither can be the other and
/// the self-edge case cannot arise.
fn weigh_comparison_axioms(
    axioms: &[SASCompareAxiom],
    weighted_graph: &mut WeightedGraph,
    predecessor_graph: &mut PredecessorGraph,
) {
    for axiom in axioms {
        let target = EitherVar::ExplicitVariable(axiom.effect);
        for &part in &axiom.parts {
            let source = EitherVar::NumericVariable(part);
            assert_ne!(source, target);
            add_edge(weighted_graph, predecessor_graph, source, target);
        }
    }
}

fn weigh_numeric_axioms(
    axioms: &[SASNumericAxiom],
    weighted_graph: &mut WeightedGraph,
    predecessor_graph: &mut PredecessorGraph,
) {
    for axiom in axioms {
        let target = EitherVar::NumericVariable(axiom.effect);
        for &part in &axiom.parts {
            let source = EitherVar::NumericVariable(part);
            add_edge(weighted_graph, predecessor_graph, source, target);
        }
    }
}

/// The graph's nodes numbered consecutively, propositional variables first, as
/// [`Scc`] and [`MaxDag`] want them.
fn node_index(var: EitherVar, num_vars: usize) -> usize {
    match var {
        EitherVar::ExplicitVariable(var) => var,
        EitherVar::NumericVariable(var) => num_vars + var,
    }
}

fn strongly_connected_components(
    num_vars: usize,
    num_numeric_vars: usize,
    weighted_graph: &WeightedGraph,
) -> Partition {
    let mut unweighted_graph: Vec<Vec<usize>> = vec![Vec::new(); num_vars + num_numeric_vars];
    for (node, successors) in weighted_graph {
        unweighted_graph[node_index(*node, num_vars)]
            .extend(successors.keys().map(|&succ| node_index(succ, num_vars)));
    }

    Scc::new(unweighted_graph)
        .get_result()
        .into_iter()
        .map(|component| {
            component
                .into_iter()
                .map(|node| {
                    if node < num_vars {
                        EitherVar::ExplicitVariable(node)
                    } else {
                        EitherVar::NumericVariable(node - num_vars)
                    }
                })
                .collect()
        })
        .collect()
}

/// Orders the components, and the variables inside a component whose
/// dependencies are cyclic by the cheapest set of dependencies to violate.
/// Goal variables are pushed to the end of their component by pricing the
/// edges into them out of reach.
fn topological_pseudo_sort(
    task: &PreprocessedTask,
    weighted_graph: &WeightedGraph,
    sccs: &Partition,
) -> Vec<EitherVar> {
    const GOAL_EDGE_SURCHARGE: u64 = 100_000;

    let mut is_goal_var = vec![false; task.vars.len()];
    for &(var, _) in &task.sas.goal.pairs {
        is_goal_var[var] = true;
    }

    let mut ordering: Vec<EitherVar> =
        Vec::with_capacity(task.vars.len() + task.numeric_vars.len());
    for component in sccs {
        if component.len() == 1 {
            ordering.push(component[0]);
            continue;
        }

        let mut variable_to_index: BTreeMap<EitherVar, usize> = BTreeMap::new();
        for (index, &var) in component.iter().enumerate() {
            variable_to_index.insert(var, index);
        }

        let subgraph: Vec<Vec<(usize, u64)>> = component
            .iter()
            .map(|var| {
                let successors = weighted_graph
                    .get(var)
                    .expect("every variable of the component is a node of the graph");
                let mut edges: Vec<(usize, u64)> = Vec::new();
                for (target, &cost) in successors {
                    let Some(&index) = variable_to_index.get(target) else {
                        continue;
                    };
                    if let EitherVar::ExplicitVariable(target) = *target
                        && is_goal_var[target]
                    {
                        edges.push((index, GOAL_EDGE_SURCHARGE + cost));
                    }
                    edges.push((index, cost));
                }
                edges
            })
            .collect();

        ordering.extend(
            MaxDag::new(subgraph)
                .get_result()
                .into_iter()
                .map(|index| component[index]),
        );
    }

    ordering
}

/// Marks everything `from` transitively depends on as necessary.
fn mark_predecessors_necessary(
    predecessor_graph: &PredecessorGraph,
    from: EitherVar,
    vars: &mut [VarState],
    numeric_vars: &mut [NumericVarState],
) {
    let mut stack = vec![from];
    while let Some(node) = stack.pop() {
        let Some(predecessors) = predecessor_graph.get(&node) else {
            continue;
        };
        for &predecessor in predecessors.keys() {
            match predecessor {
                EitherVar::ExplicitVariable(var) => {
                    if vars[var].is_necessary() {
                        continue;
                    }
                    vars[var].mark_necessary();
                    debug!("var{var} is necessary.");
                }
                EitherVar::NumericVariable(var) => {
                    if numeric_vars[var].is_necessary() {
                        continue;
                    }
                    numeric_vars[var].mark_necessary();
                    debug!("numeric var{var} is necessary.");
                }
            }
            stack.push(predecessor);
        }
    }
}

/// Marks a numeric variable the metric is computed from, and the terms it is
/// computed from in turn, as necessary. Such a variable is only read by the
/// bookkeeping of the plan's cost, never by an operator's precondition, which
/// is what makes it instrumentation rather than part of the state.
fn mark_instrumentation_necessary(
    numeric_axioms: &[SASNumericAxiom],
    numeric_vars: &mut [NumericVarState],
    var: usize,
) {
    if !numeric_vars[var].is_necessary() {
        debug!("numeric var{var} is necessary for the metric");
        numeric_vars[var].mark_instrumentation();
    }
    for axiom in numeric_axioms {
        if axiom.effect == var {
            for &part in &axiom.parts {
                mark_instrumentation_necessary(numeric_axioms, numeric_vars, part);
            }
        }
    }
}

/// An operator whose only effects are on instrumentation cannot change the
/// state the search explores, so nothing is lost by dropping it.
fn is_redundant_operator(op: &SASOperator, numeric_vars: &[NumericVarState]) -> bool {
    if !op.pre_post.is_empty() {
        return false;
    }
    for &(var, ..) in &op.assign_effects {
        if numeric_vars[var].ntype() == NumType::Regular {
            debug!(
                "Operator {} is not redundant because of its effect on numeric var{var}",
                op.name
            );
            return false;
        }
    }
    debug!("Operator {} is redundant", op.name);
    true
}
