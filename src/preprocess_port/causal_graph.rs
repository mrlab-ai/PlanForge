use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::io::Write;

use crate::preprocess_port::axiom::{
    AxiomFunctionalComparison, AxiomNumericComputation, AxiomRelational,
};
use crate::preprocess_port::helper_functions::{GlobalConstraint, DEBUG};
use crate::preprocess_port::max_dag::MaxDag;
use crate::preprocess_port::operator::Operator;
use crate::preprocess_port::scc::Scc;
use crate::preprocess_port::variable::{NumType, NumericVariable, Variable};

#[derive(Clone, Copy, Debug)]
pub struct CGVar {
    pub var: *mut Variable,
    pub nvar: *mut NumericVariable,
    pub numeric: bool,
}

impl CGVar {
    pub fn new_var(var: *mut Variable) -> Self {
        Self {
            var,
            nvar: std::ptr::null_mut(),
            numeric: false,
        }
    }

    pub fn new_nvar(nvar: *mut NumericVariable) -> Self {
        Self {
            var: std::ptr::null_mut(),
            nvar,
            numeric: true,
        }
    }

    pub fn get_name(&self) -> String {
        if self.numeric {
            assert!(!self.nvar.is_null());
            unsafe { &*self.nvar }.get_name()
        } else {
            assert!(!self.var.is_null());
            unsafe { &*self.var }.get_name()
        }
    }

    pub fn set_necessary(&self) {
        if self.numeric {
            assert!(!self.nvar.is_null());
            unsafe { &mut *self.nvar }.set_necessary();
        } else {
            assert!(!self.var.is_null());
            unsafe { &mut *self.var }.set_necessary();
        }
    }

    pub fn is_necessary(&self) -> bool {
        if self.numeric {
            assert!(!self.nvar.is_null());
            unsafe { &*self.nvar }.is_necessary()
        } else {
            assert!(!self.var.is_null());
            unsafe { &*self.var }.is_necessary()
        }
    }
}

impl PartialEq for CGVar {
    fn eq(&self, other: &Self) -> bool {
        self.numeric == other.numeric && self.var == other.var && self.nvar == other.nvar
    }
}

impl Eq for CGVar {}

impl PartialOrd for CGVar {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for CGVar {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.numeric == other.numeric {
            if self.numeric {
                (self.nvar as usize).cmp(&(other.nvar as usize))
            } else {
                (self.var as usize).cmp(&(other.var as usize))
            }
        } else if self.numeric {
            Ordering::Greater
        } else {
            Ordering::Less
        }
    }
}

pub type WeightedSuccessors = BTreeMap<CGVar, i32>;
pub type WeightedGraph = BTreeMap<CGVar, WeightedSuccessors>;
pub type Predecessors = BTreeMap<CGVar, i32>;
pub type PredecessorGraph = BTreeMap<CGVar, Predecessors>;

pub type Partition = Vec<Vec<CGVar>>;
pub type OrderingVars = Vec<CGVar>;

pub static mut G_DO_NOT_PRUNE_VARIABLES: bool = false;

#[derive(Debug)]
pub struct CausalGraph<'a> {
    variables: Vec<*mut Variable>,
    numeric_variables: Vec<*mut NumericVariable>,
    operators: &'a [Operator],
    axioms: &'a [AxiomRelational],
    ass_axioms: &'a [AxiomNumericComputation],
    comp_axioms: &'a [AxiomFunctionalComparison],
    goals: &'a Vec<(*mut Variable, i32)>,
    global_constraint: GlobalConstraint,
    metric_var: *mut NumericVariable,

    weighted_graph: WeightedGraph,
    predecessor_graph: PredecessorGraph,

    ordering: OrderingVars,
    propositional_ordering: Vec<*mut Variable>,
    numeric_ordering: Vec<*mut NumericVariable>,
    acyclic: bool,
}

impl<'a> CausalGraph<'a> {
    pub fn new(
        variables: &mut [*mut Variable],
        numeric_variables: &mut [*mut NumericVariable],
        operators: &'a [Operator],
        axioms: &'a [AxiomRelational],
        ass_axioms: &'a [AxiomNumericComputation],
        comp_axioms: &'a [AxiomFunctionalComparison],
        goals: &'a Vec<(*mut Variable, i32)>,
        global_constraint: GlobalConstraint,
        metric_var: *mut NumericVariable,
    ) -> Self {
        let mut weighted_graph: WeightedGraph = BTreeMap::new();
        for var in variables.iter_mut() {
            weighted_graph.insert(CGVar::new_var(*var), BTreeMap::new());
        }
        for nvar in numeric_variables.iter_mut() {
            weighted_graph.insert(CGVar::new_nvar(*nvar), BTreeMap::new());
        }

        let mut cg = Self {
            variables: variables.to_vec(),
            numeric_variables: numeric_variables.to_vec(),
            operators,
            axioms,
            ass_axioms,
            comp_axioms,
            goals,
            global_constraint,
            metric_var,
            weighted_graph,
            predecessor_graph: BTreeMap::new(),
            ordering: Vec::new(),
            propositional_ordering: Vec::new(),
            numeric_ordering: Vec::new(),
            acyclic: false,
        };

        cg.weigh_graph_from_ops(operators, goals);
        cg.weigh_graph_from_axioms(axioms, goals);
        cg.weigh_graph_from_comp_axioms(comp_axioms);
        cg.weigh_graph_from_ass_axioms(ass_axioms);

        let mut sccs: Partition = Vec::new();
        cg.get_strongly_connected_components(&mut sccs);

        println!(
            "The causal graph is {}acyclic.",
            if sccs.len() == (variables.len() + numeric_variables.len()) {
                ""
            } else {
                "not "
            }
        );

        cg.calculate_topological_pseudo_sort(&sccs);
        cg.calculate_important_vars();

        cg
    }

    fn weigh_graph_from_ops(
        &mut self,
        operators: &'a [Operator],
        _goals: &'a Vec<(*mut Variable, i32)>,
    ) {
        for op in operators {
            let prevail = op.get_prevail();
            let pre_posts = op.get_pre_post();
            let ass_effs = op.get_num_eff();
            let mut source_vars: Vec<CGVar> = Vec::new();
            for prev in prevail {
                source_vars.push(CGVar::new_var(prev.var as *mut Variable));
            }
            for pre_post in pre_posts {
                if pre_post.pre != -1 {
                    source_vars.push(CGVar::new_var(pre_post.var as *mut Variable));
                }
            }

            for pre_post in pre_posts {
                let curr_target = CGVar::new_var(pre_post.var as *mut Variable);
                if pre_post.is_conditional_effect {
                    for eff_cond in &pre_post.effect_conds {
                        source_vars.push(CGVar::new_var(eff_cond.var as *mut Variable));
                    }
                }

                for curr_source in &source_vars {
                    let weighted_succ = self.weighted_graph.entry(*curr_source).or_default();

                    if !self.predecessor_graph.contains_key(&curr_target) {
                        self.predecessor_graph.insert(curr_target, BTreeMap::new());
                    }
                    if *curr_source != curr_target {
                        let entry = weighted_succ.entry(curr_target).or_insert(0);
                        *entry += 1;
                        let pred = self.predecessor_graph.get_mut(&curr_target).unwrap();
                        let pred_entry = pred.entry(*curr_source).or_insert(0);
                        *pred_entry += 1;
                    }
                }

                if pre_post.is_conditional_effect {
                    let len = source_vars.len();
                    let remove_count = pre_post.effect_conds.len();
                    source_vars.truncate(len - remove_count);
                }
            }

            for ass_eff in ass_effs {
                let curr_target = CGVar::new_nvar(ass_eff.var as *mut NumericVariable);
                if ass_eff.is_conditional_effect {
                    for eff_cond in &ass_eff.effect_conds {
                        source_vars.push(CGVar::new_var(eff_cond.var as *mut Variable));
                    }
                }
                source_vars.push(CGVar::new_nvar(ass_eff.foperand as *mut NumericVariable));

                for curr_source in &source_vars {
                    let weighted_succ = self.weighted_graph.entry(*curr_source).or_default();
                    if !self.predecessor_graph.contains_key(&curr_target) {
                        self.predecessor_graph.insert(curr_target, BTreeMap::new());
                    }
                    if *curr_source != curr_target {
                        let entry = weighted_succ.entry(curr_target).or_insert(0);
                        *entry += 1;
                        let pred = self.predecessor_graph.get_mut(&curr_target).unwrap();
                        let pred_entry = pred.entry(*curr_source).or_insert(0);
                        *pred_entry += 1;
                    }
                }
                let len = source_vars.len();
                let remove_count = ass_eff.effect_conds.len() + 1;
                source_vars.truncate(len - remove_count);
            }
        }
    }

    fn weigh_graph_from_axioms(
        &mut self,
        axioms: &'a [AxiomRelational],
        _goals: &'a Vec<(*mut Variable, i32)>,
    ) {
        for axiom in axioms {
            let conds = axiom.get_conditions();
            let mut source_vars: Vec<CGVar> = Vec::new();
            for cond in conds {
                source_vars.push(CGVar::new_var(cond.var as *mut Variable));
            }
            for curr_source in &source_vars {
                let weighted_succ = self.weighted_graph.entry(*curr_source).or_default();
                let curr_target = CGVar::new_var(axiom.get_effect_var() as *mut Variable);
                if !self.predecessor_graph.contains_key(&curr_target) {
                    self.predecessor_graph.insert(curr_target, BTreeMap::new());
                }
                if *curr_source != curr_target {
                    let entry = weighted_succ.entry(curr_target).or_insert(0);
                    *entry += 1;
                    let pred = self.predecessor_graph.get_mut(&curr_target).unwrap();
                    let pred_entry = pred.entry(*curr_source).or_insert(0);
                    *pred_entry += 1;
                }
            }
        }
    }

    fn weigh_graph_from_comp_axioms(&mut self, comp_axioms: &'a [AxiomFunctionalComparison]) {
        for cax in comp_axioms {
            let mut source_vars: Vec<CGVar> = Vec::new();
            source_vars.push(CGVar::new_nvar(cax.get_left_var() as *mut NumericVariable));
            source_vars.push(CGVar::new_nvar(cax.get_right_var() as *mut NumericVariable));
            let target = CGVar::new_var(cax.get_effect_var() as *mut Variable);

            for curr_source in &source_vars {
                let weighted_succ = self.weighted_graph.entry(*curr_source).or_default();
                if !self.predecessor_graph.contains_key(&target) {
                    self.predecessor_graph.insert(target, BTreeMap::new());
                }
                let entry = weighted_succ.entry(target).or_insert(0);
                *entry += 1;
                let pred = self.predecessor_graph.get_mut(&target).unwrap();
                let pred_entry = pred.entry(*curr_source).or_insert(0);
                *pred_entry += 1;
            }
        }
    }

    fn weigh_graph_from_ass_axioms(&mut self, ass_axioms: &'a [AxiomNumericComputation]) {
        for ass_ax in ass_axioms {
            let mut source_vars: Vec<CGVar> = Vec::new();
            source_vars.push(CGVar::new_nvar(
                ass_ax.get_left_var() as *mut NumericVariable
            ));
            source_vars.push(CGVar::new_nvar(
                ass_ax.get_right_var() as *mut NumericVariable
            ));
            let target = CGVar::new_nvar(ass_ax.get_effect_var() as *mut NumericVariable);

            for curr_source in &source_vars {
                let weighted_succ = self.weighted_graph.entry(*curr_source).or_default();
                if !self.predecessor_graph.contains_key(&target) {
                    self.predecessor_graph.insert(target, BTreeMap::new());
                }
                if *curr_source != target {
                    let entry = weighted_succ.entry(target).or_insert(0);
                    *entry += 1;
                    let pred = self.predecessor_graph.get_mut(&target).unwrap();
                    let pred_entry = pred.entry(*curr_source).or_insert(0);
                    *pred_entry += 1;
                }
            }
        }
    }

    fn get_strongly_connected_components(&mut self, result: &mut Partition) {
        let mut variable_to_index: BTreeMap<CGVar, i32> = BTreeMap::new();
        let num_vars = self.variables.len();
        for (i, var) in self.variables.iter().enumerate() {
            variable_to_index.insert(CGVar::new_var(*var), i as i32);
        }
        let num_numeric_vars = self.numeric_variables.len();
        for (i, nvar) in self.numeric_variables.iter().enumerate() {
            variable_to_index.insert(CGVar::new_nvar(*nvar), (num_vars + i) as i32);
        }

        let mut unweighted_graph: Vec<Vec<i32>> = vec![Vec::new(); num_vars + num_numeric_vars];
        for (weighted_node, weighted_succ) in &self.weighted_graph {
            let index = *variable_to_index.get(weighted_node).unwrap() as usize;
            let succ = &mut unweighted_graph[index];
            for (weighted_succ_node, _) in weighted_succ {
                succ.push(*variable_to_index.get(weighted_succ_node).unwrap());
            }
        }

        let int_result = Scc::new(unweighted_graph).get_result();
        result.clear();
        for int_component in int_result {
            let mut component: Vec<CGVar> = Vec::new();
            for var_id in int_component {
                if var_id < num_vars as i32 {
                    component.push(CGVar::new_var(self.variables[var_id as usize]));
                } else {
                    let idx = (var_id as usize) - num_vars;
                    component.push(CGVar::new_nvar(self.numeric_variables[idx]));
                }
            }
            result.push(component);
        }
    }

    fn calculate_topological_pseudo_sort(&mut self, sccs: &Partition) {
        let mut goal_map: BTreeMap<*mut Variable, i32> = BTreeMap::new();
        for goal in self.goals.iter() {
            goal_map.insert(goal.0, goal.1);
        }
        for curr_scc in sccs {
            let num_scc_vars = curr_scc.len();
            if num_scc_vars > 1 {
                let mut variable_to_index: BTreeMap<CGVar, i32> = BTreeMap::new();
                for (i, v) in curr_scc.iter().enumerate() {
                    variable_to_index.insert(*v, i as i32);
                }

                let mut subgraph: Vec<Vec<(i32, i32)>> = Vec::new();
                for i in 0..num_scc_vars {
                    let all_edges = self.weighted_graph.get(&curr_scc[i]).unwrap();
                    let mut subgraph_edges: Vec<(i32, i32)> = Vec::new();
                    for (target, cost) in all_edges {
                        if let Some(index_it) = variable_to_index.get(target) {
                            let new_index = *index_it;
                            if !target.numeric {
                                if goal_map.contains_key(&(target.var as *mut Variable)) {
                                    subgraph_edges.push((new_index, 100000 + *cost));
                                }
                            }
                            subgraph_edges.push((new_index, *cost));
                        }
                    }
                    subgraph.push(subgraph_edges);
                }

                let order = MaxDag::new(subgraph).get_result();
                for i in order {
                    self.ordering.push(curr_scc[i as usize]);
                }
            } else {
                self.ordering.push(curr_scc[0]);
            }
        }
    }

    fn calculate_important_vars(&mut self) {
        for goal in self.goals.iter() {
            let var = unsafe { &mut *goal.0 };
            if !var.is_necessary() {
                if DEBUG {
                    println!("var {} is directly neccessary (goal).", var.get_name());
                }
                var.set_necessary();
                self.dfs(CGVar::new_var(goal.0));
            }
        }

        let gc_var = unsafe { &mut *self.global_constraint.var };
        if !gc_var.is_necessary() {
            if DEBUG {
                println!(
                    "var {} is directly neccessary (global constraint).",
                    gc_var.get_name()
                );
            }
            gc_var.set_necessary();
            self.dfs(CGVar::new_var(self.global_constraint.var));
        }

        self.set_variable_instrumentation_necessary(self.metric_var);
        for op in self.operators {
            for num_eff in op.get_num_eff() {
                let var = unsafe { &*num_eff.var };
                if var.get_type() == NumType::Instrumentation {
                    assert!(var.is_necessary());
                    self.set_variable_instrumentation_necessary(
                        num_eff.foperand as *mut NumericVariable,
                    );
                }
            }
        }

        assert!(self.propositional_ordering.is_empty());
        assert!(self.numeric_ordering.is_empty());
        assert!(self.ordering.len() == self.numeric_variables.len() + self.variables.len());
        for cg_var in &self.ordering {
            if cg_var.numeric {
                let nvar = unsafe { &mut *cg_var.nvar };
                if nvar.is_necessary() || unsafe { G_DO_NOT_PRUNE_VARIABLES } {
                    self.numeric_ordering.push(cg_var.nvar);
                }
            } else {
                let var = unsafe { &mut *cg_var.var };
                if var.is_necessary() || unsafe { G_DO_NOT_PRUNE_VARIABLES } {
                    self.propositional_ordering.push(cg_var.var);
                }
            }
        }
        for (i, var) in self.propositional_ordering.iter().enumerate() {
            unsafe { &mut **var }.set_level(i as i32);
        }
        for (i, nvar) in self.numeric_ordering.iter().enumerate() {
            unsafe { &mut **nvar }.set_level(i as i32);
        }
        println!(
            "{} variables of {} necessary",
            self.propositional_ordering.len(),
            self.variables.len()
        );
        println!(
            "{} numeric variables of {} necessary",
            self.numeric_ordering.len(),
            self.numeric_variables.len()
        );
    }

    fn dfs(&self, from: CGVar) {
        if let Some(preds) = self.predecessor_graph.get(&from) {
            for (pred, _) in preds {
                let curr_predecessor = *pred;
                if !curr_predecessor.is_necessary() {
                    curr_predecessor.set_necessary();
                    if DEBUG {
                        println!("var {} is neccessary.", curr_predecessor.get_name());
                    }
                    self.dfs(curr_predecessor);
                }
            }
        }
    }

    fn set_variable_instrumentation_necessary(&self, inst_var: *mut NumericVariable) {
        let var = unsafe { &mut *inst_var };
        if !var.is_necessary() {
            if DEBUG {
                println!("{} is necessary for the metric", var.get_name());
            }
            var.set_instrumentation();
        }
        for ass_ax in self.ass_axioms {
            if inst_var == ass_ax.get_effect_var() as *mut NumericVariable {
                self.set_variable_instrumentation_necessary(
                    ass_ax.get_left_var() as *mut NumericVariable
                );
                self.set_variable_instrumentation_necessary(
                    ass_ax.get_right_var() as *mut NumericVariable
                );
            }
        }
    }

    pub fn get_metric_index(&self) -> i32 {
        unsafe { &*self.metric_var }.get_level()
    }

    pub fn get_variable_ordering(&self) -> &Vec<*mut Variable> {
        &self.propositional_ordering
    }

    pub fn get_numeric_variable_ordering(&self) -> &Vec<*mut NumericVariable> {
        &self.numeric_ordering
    }

    pub fn is_acyclic(&self) -> bool {
        self.acyclic
    }

    pub fn dump(&self) {
        for (source, succs) in &self.weighted_graph {
            println!("dependent on var {}: ", source.get_name());
            for (succ, weight) in succs {
                println!("  [{}, {}]", succ.get_name(), weight);
            }
        }
        for (source, succs) in &self.predecessor_graph {
            println!("var {} is dependent of: ", source.get_name());
            for (succ, weight) in succs {
                println!("  [{}, {}]", succ.get_name(), weight);
            }
        }
    }

    pub fn generate_cpp_input<W: Write>(&self, out: &mut W, ordered_vars: &Vec<*mut Variable>) {
        let mut succs: Vec<Option<WeightedSuccessors>> = vec![None; ordered_vars.len()];
        let mut number_of_succ: Vec<i32> = vec![0; ordered_vars.len()];
        for (source, succ) in &self.weighted_graph {
            if !source.numeric {
                let source_var = unsafe { &*source.var };
                if source_var.get_level() != -1 {
                    let mut num = 0;
                    for (succ_var, _) in succ {
                        if !succ_var.numeric {
                            let var = unsafe { &*succ_var.var };
                            if var.get_level() != -1 {
                                num += 1;
                            }
                        }
                    }
                    succs[source_var.get_level() as usize] = Some(succ.clone());
                    number_of_succ[source_var.get_level() as usize] = num;
                }
            }
        }
        let num_vars = ordered_vars.len();
        for i in 0..num_vars {
            let curr = succs[i].clone().unwrap_or_default();
            writeln!(out, "{}", number_of_succ[i]).unwrap();
            for (succ, weight) in &curr {
                if !succ.numeric {
                    let var = unsafe { &*succ.var };
                    if var.get_level() != -1 {
                        writeln!(out, "{} {}", var.get_level(), weight).unwrap();
                    }
                }
            }
        }
    }
}
