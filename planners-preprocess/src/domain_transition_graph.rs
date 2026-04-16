use std::cmp::Ordering;
use std::io::Write;

use crate::Condition;
use crate::axiom::AxiomRelational;
use crate::fact::ExplicitFact;
use crate::operator::{Operator, PrePost, Prevail};
use crate::scc::Scc;
use crate::variable::ExplicitVariable;

#[derive(Debug, Clone)]
struct Transition {
    target: usize,
    op: usize,
    cost: f64,
    condition: Condition,
}

impl Transition {
    fn new(target: usize, op: usize) -> Self {
        Self {
            target,
            op,
            cost: 0.0,
            condition: Vec::new(),
        }
    }
}

impl PartialEq for Transition {
    fn eq(&self, other: &Self) -> bool {
        self.target == other.target && self.op == other.op && self.condition == other.condition
    }
}

impl Eq for Transition {}

impl PartialOrd for Transition {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Transition {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.target != other.target {
            self.target.cmp(&other.target)
        } else if self.condition.len() != other.condition.len() {
            self.condition.len().cmp(&other.condition.len())
        } else {
            self.cost
                .partial_cmp(&other.cost)
                .unwrap_or(Ordering::Equal)
        }
    }
}

type Vertex = Vec<Transition>;

#[derive(Debug, Clone)]
pub struct DomainTransitionGraph {
    vertices: Vec<Vertex>,
    level: i32,
}

impl DomainTransitionGraph {
    pub fn new(var: &ExplicitVariable) -> Self {
        let mut vertices: Vec<Vertex> = Vec::new();
        let range = var.get_range();
        vertices.resize(range, Vec::new());
        let level = var.get_level();
        assert!(level != -1);
        Self { vertices, level }
    }

    pub fn add_transition(
        &mut self,
        from: usize,
        to: usize,
        op: &Operator,
        op_index: usize,
        pre_post: &PrePost,
        vars: &[ExplicitVariable],
    ) {
        let var = pre_post.var;
        assert!(vars[var].get_level() == self.level && pre_post.post == to);
        let mut trans = Transition::new(to, op_index);
        trans.cost = op.get_cost();
        let cond = &mut trans.condition;

        let prevail = op.get_prevail();
        for Prevail { var, prev } in prevail {
            cond.push(ExplicitFact {
                var: *var,
                value: *prev,
            });
        }
        for op_pre_post in op.get_pre_post() {
            if let Some(pre) = op_pre_post.pre
                && vars[op_pre_post.var].get_level() != self.level
            {
                cond.push(ExplicitFact {
                    var: op_pre_post.var,
                    value: pre,
                });
            }
        }

        for eff_cond in &pre_post.effect_conds {
            if vars[eff_cond.var].get_level() == self.level {
                if eff_cond.cond != from {
                    return;
                }
            } else {
                trans.condition.push(ExplicitFact {
                    var: eff_cond.var,
                    value: eff_cond.cond,
                });
            }
        }

        self.vertices[from].push(trans);
    }

    pub fn add_ax_transition(
        &mut self,
        from: usize,
        to: usize,
        ax: &AxiomRelational,
        ax_index: usize,
    ) {
        let mut trans = Transition::new(to, ax_index);
        let cond = &mut trans.condition;
        for ax_cond in ax.get_conditions() {
            cond.push(ExplicitFact {
                var: ax_cond.var,
                value: ax_cond.cond,
            });
        }
        self.vertices[from].push(trans);
    }

    pub fn finalize(&mut self) {
        for transitions in &mut self.vertices {
            transitions.sort();
            transitions.dedup();
            for trans in transitions.iter_mut() {
                trans.condition.sort();
            }

            let mut undominated_trans: Vec<Transition> = Vec::new();
            let mut is_dominated: Vec<bool> = vec![false; transitions.len()];
            let num_transitions = transitions.len();
            for j in 0..num_transitions {
                if !is_dominated[j] {
                    let trans = transitions[j].clone();
                    undominated_trans.push(trans.clone());
                    let cond = trans.condition.clone();
                    let mut comp = j + 1;
                    while comp < num_transitions {
                        if is_dominated[comp] {
                            comp += 1;
                            continue;
                        }
                        let other_trans = &transitions[comp];
                        if other_trans.cost < trans.cost {
                            comp += 1;
                            continue;
                        }
                        assert!(other_trans.target >= trans.target);
                        if other_trans.target != trans.target {
                            break;
                        } else {
                            assert!(other_trans.condition.len() >= cond.len());
                            if cond.is_empty() {
                                is_dominated[comp] = true;
                                comp += 1;
                            } else {
                                let mut same_conditions = true;
                                for c1 in &cond {
                                    let mut comp_dominated = false;
                                    for c2 in &other_trans.condition {
                                        if c2.var > c1.var {
                                            break;
                                        }
                                        if c2.var == c1.var && c2.value == c1.value {
                                            comp_dominated = true;
                                            break;
                                        }
                                    }
                                    if !comp_dominated {
                                        same_conditions = false;
                                        break;
                                    }
                                }
                                is_dominated[comp] = same_conditions;
                                comp += 1;
                            }
                        }
                    }
                }
            }
            *transitions = undominated_trans;
        }
    }

    pub fn dump(&self, vars: &[ExplicitVariable]) {
        println!("Level: {}", self.level);
        let num_vertices = self.vertices.len();
        for i in 0..num_vertices {
            println!("  From value {}:", i);
            for trans in &self.vertices[i] {
                println!("    To value {}", trans.target);
                for cond in &trans.condition {
                    println!("      if {} = {}", vars[cond.var].get_name(), cond.value);
                }
            }
        }
    }

    pub fn to_sas<W: Write>(&self, out: &mut W, vars: &[ExplicitVariable]) {
        for vertex in &self.vertices {
            writeln!(out, "{}", vertex.len()).unwrap();
            for trans in vertex {
                writeln!(out, "{}", trans.target).unwrap();
                writeln!(out, "{}", trans.op).unwrap();
                let mut number = 0;
                for cond in &trans.condition {
                    if vars[cond.var].get_level() != -1 {
                        number += 1;
                    }
                }
                writeln!(out, "{}", number).unwrap();
                for cond in &trans.condition {
                    if vars[cond.var].get_level() != -1 {
                        writeln!(out, "{} {}", vars[cond.var].get_level(), cond.value).unwrap();
                    }
                }
            }
        }
    }

    pub fn is_strongly_connected(&self) -> bool {
        let mut easy_graph: Vec<Vec<usize>> = Vec::new();
        let num_vertices = self.vertices.len();
        for i in 0..num_vertices {
            let mut edges: Vec<usize> = Vec::new();
            for trans in &self.vertices[i] {
                edges.push(trans.target);
            }
            easy_graph.push(edges);
        }
        let sccs = Scc::new(easy_graph).get_result();

        sccs.len() == 1
    }
}

pub fn build_dtgs(
    ordered_variables: &[ExplicitVariable],
    orig_variables: &[ExplicitVariable],
    operators: &[Operator],
    axioms: &[AxiomRelational],
) -> Vec<DomainTransitionGraph> {
    let mut transition_graphs = Vec::with_capacity(ordered_variables.len());
    for var in ordered_variables {
        transition_graphs.push(DomainTransitionGraph::new(var));
    }
    for (i, op) in operators.iter().enumerate() {
        for eff in op.get_pre_post() {
            let var_level = orig_variables[eff.var].get_level();
            if var_level != -1 {
                let pre = eff.pre;
                let post = eff.post;
                if let Some(pre_var) = pre {
                    transition_graphs[var_level as usize].add_transition(
                        pre_var,
                        post,
                        op,
                        i,
                        eff,
                        orig_variables,
                    );
                } else {
                    for pre_val in 0..orig_variables[eff.var].get_range() {
                        if pre_val != post {
                            transition_graphs[var_level as usize].add_transition(
                                pre_val,
                                post,
                                op,
                                i,
                                eff,
                                orig_variables,
                            );
                        }
                    }
                }
            }
        }
    }
    for (i, ax) in axioms.iter().enumerate() {
        let var = ax.get_effect_var();
        let var_level = orig_variables[var].get_level();
        assert!(var_level != -1);
        let old_val = ax.get_old_val();
        let new_val = ax.get_effect_val();
        transition_graphs[var_level as usize].add_ax_transition(old_val, new_val, ax, i);
    }
    for transition_graph in transition_graphs.iter_mut() {
        transition_graph.finalize();
    }

    transition_graphs
}

#[allow(clippy::needless_range_loop)]
pub fn are_dtgs_strongly_connected(transition_graphs: &[DomainTransitionGraph]) -> bool {
    let mut connected = true;
    let num_dtgs = transition_graphs.len();
    for i in 0..num_dtgs.saturating_sub(1) {
        if !transition_graphs[i].is_strongly_connected() {
            connected = false;
            break;
        }
    }
    connected
}
