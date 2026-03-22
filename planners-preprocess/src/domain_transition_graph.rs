use std::cmp::Ordering;
use std::io::Write;

use crate::axiom::AxiomRelational;
use crate::operator::{Operator, PrePost, Prevail};
use crate::scc::Scc;
use crate::variable::Variable;

pub type Condition = Vec<(*const Variable, i32)>;

#[derive(Debug, Clone)]
struct Transition {
    target: i32,
    op: i32,
    cost: f64,
    condition: Condition,
}

impl Transition {
    fn new(target: i32, op: i32) -> Self {
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
    pub fn new(var: &Variable) -> Self {
        let mut vertices: Vec<Vertex> = Vec::new();
        let range = var.get_range();
        vertices.resize(range as usize, Vec::new());
        let level = var.get_level();
        assert!(level != -1);
        Self { vertices, level }
    }

    pub fn add_transition(
        &mut self,
        from: i32,
        to: i32,
        op: &Operator,
        op_index: i32,
        pre_post: &PrePost,
    ) {
        let var = unsafe { &*pre_post.var };
        assert!(var.get_level() == self.level && pre_post.post == to);
        let mut trans = Transition::new(to, op_index);
        trans.cost = op.get_cost();
        let cond = &mut trans.condition;

        let prevail = op.get_prevail();
        for Prevail { var, prev } in prevail {
            cond.push((*var, *prev));
        }
        for op_pre_post in op.get_pre_post() {
            if op_pre_post.pre != -1 {
                let pp_var = unsafe { &*op_pre_post.var };
                if pp_var.get_level() == self.level {
                    if op_pre_post.pre != from {
                        continue;
                    }
                } else {
                    cond.push((op_pre_post.var, op_pre_post.pre));
                }
            }
        }

        for eff_cond in &pre_post.effect_conds {
            let eff_var = unsafe { &*eff_cond.var };
            if eff_var.get_level() == self.level {
                if eff_cond.cond != from {
                    return;
                }
            } else {
                trans.condition.push((eff_cond.var, eff_cond.cond));
            }
        }

        self.vertices[from as usize].push(trans);
    }

    pub fn add_ax_transition(&mut self, from: i32, to: i32, ax: &AxiomRelational, ax_index: i32) {
        let mut trans = Transition::new(to, ax_index);
        let cond = &mut trans.condition;
        for ax_cond in ax.get_conditions() {
            cond.push((ax_cond.var, ax_cond.cond));
        }
        self.vertices[from as usize].push(trans);
    }

    pub fn finalize(&mut self) {
        for transitions in &mut self.vertices {
            transitions.sort();
            transitions.dedup();
            for trans in transitions.iter_mut() {
                trans.condition.sort_by(|a, b| {
                    let ap = a.0 as usize;
                    let bp = b.0 as usize;
                    if ap == bp { a.1.cmp(&b.1) } else { ap.cmp(&bp) }
                });
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
                                        if (c2.0 as usize) > (c1.0 as usize) {
                                            break;
                                        }
                                        if c2.0 == c1.0 && c2.1 == c1.1 {
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

    pub fn dump(&self) {
        println!("Level: {}", self.level);
        let num_vertices = self.vertices.len();
        for i in 0..num_vertices {
            println!("  From value {}:", i);
            for trans in &self.vertices[i] {
                println!("    To value {}", trans.target);
                for cond in &trans.condition {
                    let var = unsafe { &*cond.0 };
                    println!("      if {} = {}", var.get_name(), cond.1);
                }
            }
        }
    }

    pub fn generate_cpp_input<W: Write>(&self, out: &mut W) {
        for vertex in &self.vertices {
            writeln!(out, "{}", vertex.len()).unwrap();
            for trans in vertex {
                writeln!(out, "{}", trans.target).unwrap();
                writeln!(out, "{}", trans.op).unwrap();
                let mut number = 0;
                for cond in &trans.condition {
                    let var = unsafe { &*cond.0 };
                    if var.get_level() != -1 {
                        number += 1;
                    }
                }
                writeln!(out, "{}", number).unwrap();
                for cond in &trans.condition {
                    let var = unsafe { &*cond.0 };
                    if var.get_level() != -1 {
                        writeln!(out, "{} {}", var.get_level(), cond.1).unwrap();
                    }
                }
            }
        }
    }

    pub fn is_strongly_connected(&self) -> bool {
        let mut easy_graph: Vec<Vec<i32>> = Vec::new();
        let num_vertices = self.vertices.len();
        for i in 0..num_vertices {
            let mut edges: Vec<i32> = Vec::new();
            for trans in &self.vertices[i] {
                edges.push(trans.target);
            }
            easy_graph.push(edges);
        }
        let sccs = Scc::new(easy_graph).get_result();
        let connected = sccs.len() == 1;
        connected
    }
}

pub fn build_dtgs(
    var_order: &Vec<*mut Variable>,
    operators: &[Operator],
    axioms: &[AxiomRelational],
    transition_graphs: &mut Vec<DomainTransitionGraph>,
) {
    for var in var_order {
        let var_ref = unsafe { &**var };
        transition_graphs.push(DomainTransitionGraph::new(var_ref));
    }
    for (i, op) in operators.iter().enumerate() {
        for eff in op.get_pre_post() {
            let var = unsafe { &*eff.var };
            let var_level = var.get_level();
            if var_level != -1 {
                let pre = eff.pre;
                let post = eff.post;
                if pre != -1 {
                    transition_graphs[var_level as usize]
                        .add_transition(pre, post, op, i as i32, eff);
                } else {
                    for pre_val in 0..var.get_range() {
                        if pre_val != post {
                            transition_graphs[var_level as usize]
                                .add_transition(pre_val, post, op, i as i32, eff);
                        }
                    }
                }
            }
        }
    }
    for (i, ax) in axioms.iter().enumerate() {
        let var = unsafe { &*ax.get_effect_var() };
        let var_level = var.get_level();
        assert!(var_level != -1);
        let old_val = ax.get_old_val();
        let new_val = ax.get_effect_val();
        transition_graphs[var_level as usize].add_ax_transition(old_val, new_val, ax, i as i32);
    }
    for transition_graph in transition_graphs.iter_mut() {
        transition_graph.finalize();
    }
}

pub fn are_dtgs_strongly_connected(transition_graphs: &[DomainTransitionGraph]) -> bool {
    let mut connected = true;
    let num_dtgs = transition_graphs.len();
    for i in 0..num_dtgs.saturating_sub(1) {
        if !transition_graphs[i].is_strongly_connected() {
            connected = false;
        }
    }
    connected
}
