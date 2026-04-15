use std::cell::RefCell;
use std::cmp::max;
use std::collections::BTreeMap;
use std::ops::Deref;

use crate::axiom::{AxiomFunctionalComparison, AxiomNumericComputation, AxiomRelational};
use crate::fact::ExplicitFact;
use crate::max_dag::MaxDag;
use crate::mutex_group::MutexGroup;
use crate::operator::Operator;
use crate::scc::Scc;
use crate::variable::{ExplicitVariable, NumType, NumericVariable};
use crate::{DEBUG, GlobalConstraint};

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum EitherVar {
    ExplicitVariable(usize),
    NumericVariable(usize),
}

impl EitherVar {
    pub fn get_name(&self, vars: &[ExplicitVariable], numeric_vars: &[NumericVariable]) -> String {
        match self {
            EitherVar::ExplicitVariable(v) => vars[*v].get_name(),
            EitherVar::NumericVariable(v) => numeric_vars[*v].get_name(),
        }
    }
}

impl Deref for EitherVar {
    type Target = usize;

    fn deref(&self) -> &Self::Target {
        match self {
            Self::ExplicitVariable(i) => i,
            Self::NumericVariable(i) => i,
        }
    }
}

pub type WeightedSuccessors = BTreeMap<EitherVar, u64>;
pub type WeightedGraph = BTreeMap<EitherVar, WeightedSuccessors>;
pub type Predecessors = BTreeMap<EitherVar, u64>;
pub type PredecessorGraph = BTreeMap<EitherVar, Predecessors>;

pub type Partition = Vec<Vec<EitherVar>>;
pub type OrderingVars = Vec<EitherVar>;
pub type ExplicitOrderingVars = Vec<usize>;
pub type NumericOrderingVars = Vec<usize>;

#[derive(Debug)]
pub struct CausalGraph {
    variables: RefCell<Vec<ExplicitVariable>>,
    numeric_variables: RefCell<Vec<NumericVariable>>,
    operators: Vec<Operator>,
    axioms: Vec<AxiomRelational>,
    ass_axioms: Vec<AxiomNumericComputation>,
    comp_axioms: Vec<AxiomFunctionalComparison>,
    mutexes: Vec<MutexGroup>,
    goals: Vec<ExplicitFact>,
    global_constraint: Option<GlobalConstraint>,
    metric_var: usize,
    prune_variables: bool,

    weighted_graph: WeightedGraph,
    predecessor_graph: PredecessorGraph,

    ordering: OrderingVars,
    propositional_ordering: ExplicitOrderingVars,
    numeric_ordering: NumericOrderingVars,
    acyclic: bool,
}

impl CausalGraph {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        variables: Vec<ExplicitVariable>,
        numeric_variables: Vec<NumericVariable>,
        operators: Vec<Operator>,
        axioms: Vec<AxiomRelational>,
        ass_axioms: Vec<AxiomNumericComputation>,
        comp_axioms: Vec<AxiomFunctionalComparison>,
        mutexes: Vec<MutexGroup>,
        goals: Vec<ExplicitFact>,
        global_constraint: Option<GlobalConstraint>,
        metric_var: usize,
        prune_variables: bool,
    ) -> Self {
        let mut weighted_graph: WeightedGraph = BTreeMap::new();
        for v in variables.iter() {
            weighted_graph.insert(EitherVar::ExplicitVariable(v.index), BTreeMap::new());
        }
        for v in numeric_variables.iter() {
            weighted_graph.insert(EitherVar::NumericVariable(v.index), BTreeMap::new());
        }

        let (weighted_graph, predecessor_graph) =
            Self::weigh_graph_from_ops(&operators, weighted_graph);
        let (weighted_graph, predecessor_graph) =
            Self::weigh_graph_from_axioms(&axioms, weighted_graph, predecessor_graph);
        let (weighted_graph, predecessor_graph) =
            Self::weigh_graph_from_comp_axioms(&comp_axioms, weighted_graph, predecessor_graph);
        let (weighted_graph, predecessor_graph) =
            Self::weigh_graph_from_ass_axioms(&ass_axioms, weighted_graph, predecessor_graph);

        let (sccs, weighted_graph) =
            Self::get_strongly_connected_components(&variables, &numeric_variables, weighted_graph);
        let (ordering, weighted_graph) =
            Self::calculate_topological_pseudo_sort(&goals, weighted_graph, &sccs);

        let acyclic = sccs.len() == (variables.len() + numeric_variables.len());
        println!(
            "The causal graph is {}acyclic.",
            if acyclic { "" } else { "not " }
        );

        let mut cg = Self {
            variables: RefCell::new(variables),
            numeric_variables: RefCell::new(numeric_variables),
            operators,
            axioms,
            ass_axioms,
            comp_axioms,
            mutexes,
            goals,
            global_constraint,
            metric_var,
            prune_variables,
            weighted_graph,
            predecessor_graph,
            ordering,
            propositional_ordering: Vec::new(),
            numeric_ordering: Vec::new(),
            acyclic,
        };

        cg.calculate_important_vars();

        cg
    }

    fn weigh_graph_from_ops(
        operators: &[Operator],
        mut weighted_graph: WeightedGraph,
    ) -> (WeightedGraph, PredecessorGraph) {
        let mut predecessor_graph: PredecessorGraph = BTreeMap::new();
        for op in operators.iter() {
            let prevail = op.get_prevail();
            let pre_posts = op.get_pre_post();
            let ass_effs = op.get_num_eff();
            let mut source_vars: Vec<EitherVar> = Vec::new();
            for prev in prevail.iter() {
                source_vars.push(EitherVar::ExplicitVariable(prev.var));
            }
            for pre_post in pre_posts.iter() {
                if pre_post.pre.is_some() {
                    source_vars.push(EitherVar::ExplicitVariable(pre_post.var));
                }
            }

            for pre_post in pre_posts.iter() {
                let curr_target = EitherVar::ExplicitVariable(pre_post.var);
                if pre_post.is_conditional_effect {
                    for eff_cond in &pre_post.effect_conds {
                        source_vars.push(EitherVar::ExplicitVariable(eff_cond.var));
                    }
                }

                for curr_source in &source_vars {
                    let weighted_succ = weighted_graph.entry(*curr_source).or_default();

                    predecessor_graph.entry(curr_target).or_default();
                    if *curr_source != curr_target {
                        let entry = weighted_succ.entry(curr_target).or_insert(0);
                        *entry += 1;
                        let pred = predecessor_graph.get_mut(&curr_target).unwrap();
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

            for ass_eff in ass_effs.iter() {
                let curr_target = EitherVar::NumericVariable(ass_eff.var);
                if ass_eff.is_conditional_effect {
                    for eff_cond in &ass_eff.effect_conds {
                        source_vars.push(EitherVar::ExplicitVariable(eff_cond.var));
                    }
                }
                source_vars.push(EitherVar::NumericVariable(ass_eff.foperand));

                for curr_source in &source_vars {
                    let weighted_succ = weighted_graph.entry(*curr_source).or_default();
                    predecessor_graph.entry(curr_target).or_default();
                    if *curr_source != curr_target {
                        let entry = weighted_succ.entry(curr_target).or_insert(0);
                        *entry += 1;
                        let pred = predecessor_graph.get_mut(&curr_target).unwrap();
                        let pred_entry = pred.entry(*curr_source).or_insert(0);
                        *pred_entry += 1;
                    }
                }
                let len = source_vars.len();
                let remove_count = ass_eff.effect_conds.len() + 1;
                source_vars.truncate(len - remove_count);
            }
        }

        (weighted_graph, predecessor_graph)
    }

    fn weigh_graph_from_axioms(
        axioms: &[AxiomRelational],
        mut weighted_graph: WeightedGraph,
        mut predecessor_graph: PredecessorGraph,
    ) -> (WeightedGraph, PredecessorGraph) {
        for axiom in axioms.iter() {
            for cond in axiom.get_conditions().iter() {
                let curr_source = EitherVar::ExplicitVariable(cond.var);
                let weighted_succ = weighted_graph.entry(curr_source).or_default();
                let curr_target = EitherVar::ExplicitVariable(axiom.get_effect_var());
                predecessor_graph.entry(curr_target).or_default();
                if curr_source != curr_target {
                    let entry = weighted_succ.entry(curr_target).or_insert(0);
                    *entry += 1;
                    let pred = predecessor_graph.get_mut(&curr_target).unwrap();
                    let pred_entry = pred.entry(curr_source).or_insert(0);
                    *pred_entry += 1;
                }
            }
        }

        (weighted_graph, predecessor_graph)
    }

    fn weigh_graph_from_comp_axioms(
        comp_axioms: &[AxiomFunctionalComparison],
        mut weighted_graph: WeightedGraph,
        mut predecessor_graph: PredecessorGraph,
    ) -> (WeightedGraph, PredecessorGraph) {
        for cax in comp_axioms {
            let target = EitherVar::ExplicitVariable(cax.get_effect_var());

            for curr_source in [
                EitherVar::NumericVariable(cax.get_left_var()),
                EitherVar::NumericVariable(cax.get_right_var()),
            ] {
                let weighted_succ = weighted_graph.entry(curr_source).or_default();
                predecessor_graph.entry(target).or_default();
                let entry = weighted_succ.entry(target).or_insert(0);
                *entry += 1;
                let pred = predecessor_graph.get_mut(&target).unwrap();
                let pred_entry = pred.entry(curr_source).or_insert(0);
                *pred_entry += 1;
            }
        }

        (weighted_graph, predecessor_graph)
    }

    fn weigh_graph_from_ass_axioms(
        ass_axioms: &[AxiomNumericComputation],
        mut weighted_graph: WeightedGraph,
        mut predecessor_graph: PredecessorGraph,
    ) -> (WeightedGraph, PredecessorGraph) {
        for ass_ax in ass_axioms {
            let target = EitherVar::NumericVariable(ass_ax.get_effect_var());

            for curr_source in [
                EitherVar::NumericVariable(ass_ax.get_left_var()),
                EitherVar::NumericVariable(ass_ax.get_right_var()),
            ] {
                let weighted_succ = weighted_graph.entry(curr_source).or_default();
                predecessor_graph.entry(target).or_default();
                if curr_source != target {
                    let entry = weighted_succ.entry(target).or_insert(0);
                    *entry += 1;
                    let pred = predecessor_graph.get_mut(&target).unwrap();
                    let pred_entry = pred.entry(curr_source).or_insert(0);
                    *pred_entry += 1;
                }
            }
        }

        (weighted_graph, predecessor_graph)
    }

    fn get_strongly_connected_components(
        variables: &[ExplicitVariable],
        numeric_variables: &[NumericVariable],
        weighted_graph: WeightedGraph,
    ) -> (Partition, WeightedGraph) {
        let mut result = Vec::new();
        let mut variable_to_index: BTreeMap<EitherVar, usize> = BTreeMap::new();
        let num_vars = variables.len();
        for (i, var) in variables.iter().enumerate() {
            variable_to_index.insert(EitherVar::ExplicitVariable(var.index), i);
        }
        let num_numeric_vars = numeric_variables.len();
        for (i, nvar) in numeric_variables.iter().enumerate() {
            variable_to_index.insert(EitherVar::NumericVariable(nvar.index), num_vars + i);
        }

        let mut unweighted_graph: Vec<Vec<usize>> = vec![Vec::new(); num_vars + num_numeric_vars];
        for (weighted_node, weighted_succ) in weighted_graph.iter() {
            let index = *variable_to_index.get(weighted_node).unwrap();
            let succ = &mut unweighted_graph[index];
            for weighted_succ_node in weighted_succ.keys() {
                succ.push(*variable_to_index.get(weighted_succ_node).unwrap());
            }
        }

        let int_result = Scc::new(unweighted_graph).get_result();
        for int_component in int_result {
            let mut component: Vec<EitherVar> = Vec::new();
            for var_id in int_component {
                if var_id < num_vars {
                    assert!(var_id == variables[var_id].index);
                    component.push(EitherVar::ExplicitVariable(var_id));
                } else {
                    let idx = var_id - num_vars;
                    assert!(idx == numeric_variables[idx].index);
                    component.push(EitherVar::NumericVariable(idx));
                }
            }
            result.push(component);
        }

        (result, weighted_graph)
    }

    #[allow(clippy::needless_range_loop)]
    fn calculate_topological_pseudo_sort(
        goals: &[ExplicitFact],
        weighted_graph: WeightedGraph,
        sccs: &Partition,
    ) -> (OrderingVars, WeightedGraph) {
        let mut ordering: OrderingVars = Vec::new();
        let mut goal_map: BTreeMap<usize, usize> = BTreeMap::new();
        for goal in goals.iter() {
            goal_map.insert(goal.var, goal.value);
        }
        for curr_scc in sccs {
            let num_scc_vars = curr_scc.len();
            if num_scc_vars > 1 {
                let mut variable_to_index: BTreeMap<EitherVar, usize> = BTreeMap::new();
                for (i, v) in curr_scc.iter().enumerate() {
                    variable_to_index.insert(*v, i);
                }

                let mut subgraph: Vec<Vec<(usize, u64)>> = Vec::new();
                for i in 0..num_scc_vars {
                    let all_edges = weighted_graph.get(&curr_scc[i]).unwrap();
                    let mut subgraph_edges: Vec<(usize, u64)> = Vec::new();
                    for (target, cost) in all_edges {
                        if let Some(index_it) = variable_to_index.get(target) {
                            let new_index = *index_it;
                            if let EitherVar::ExplicitVariable(v) = target
                                && goal_map.contains_key(v)
                            {
                                subgraph_edges.push((new_index, 100000 + *cost));
                            }
                            subgraph_edges.push((new_index, *cost));
                        }
                    }
                    subgraph.push(subgraph_edges);
                }

                let order = MaxDag::new(subgraph).get_result();
                for i in order {
                    ordering.push(curr_scc[i]);
                }
            } else {
                ordering.push(curr_scc[0]);
            }
        }

        (ordering, weighted_graph)
    }

    fn calculate_important_vars(&mut self) {
        let mut variables = self.variables.borrow_mut();
        let mut numeric_variables = self.numeric_variables.borrow_mut();
        for goal in self.goals.iter() {
            let var = goal.var;
            if !variables[var].is_necessary() {
                if DEBUG {
                    println!(
                        "var {} is directly necessary (goal).",
                        variables[var].get_name()
                    );
                }
                variables[var].set_necessary();
                self.dfs(
                    EitherVar::ExplicitVariable(goal.var),
                    &mut variables,
                    &mut numeric_variables,
                );
            }
        }

        if let Some(gc) = self.global_constraint {
            let gc_var = gc.var;
            if !variables[gc_var].is_necessary() {
                if DEBUG {
                    println!(
                        "var {} is directly necessary (global constraint).",
                        variables[gc_var].get_name()
                    );
                }
                variables[gc_var].set_necessary();
                self.dfs(
                    EitherVar::ExplicitVariable(gc.var),
                    &mut variables,
                    &mut numeric_variables,
                );
            }
        }

        self.set_variable_instrumentation_necessary(&mut numeric_variables, self.metric_var);
        for op in &self.operators {
            for num_eff in op.get_num_eff() {
                let var = num_eff.var;
                if numeric_variables[var].get_type() == NumType::Instrumentation {
                    assert!(numeric_variables[var].is_necessary());
                    self.set_variable_instrumentation_necessary(
                        &mut numeric_variables,
                        num_eff.foperand,
                    );
                }
            }
        }

        assert!(self.propositional_ordering.is_empty());
        assert!(self.numeric_ordering.is_empty());
        assert!(self.ordering.len() == numeric_variables.len() + variables.len());
        for cg_var in &self.ordering {
            match cg_var {
                EitherVar::ExplicitVariable(v) => {
                    if variables[*v].is_necessary() || !self.prune_variables {
                        self.propositional_ordering.push(*v);
                    }
                }
                EitherVar::NumericVariable(v) => {
                    if numeric_variables[*v].is_necessary() || !self.prune_variables {
                        self.numeric_ordering.push(*v);
                    }
                }
            }
        }
        for (i, var) in self.propositional_ordering.iter().enumerate() {
            variables[*var].set_level(i as i32);
        }
        for (i, nvar) in self.numeric_ordering.iter().enumerate() {
            numeric_variables[*nvar].set_level(i as i32);
        }
        println!(
            "{} variables of {} necessary",
            self.propositional_ordering.len(),
            variables.len()
        );
        println!(
            "{} numeric variables of {} necessary",
            self.numeric_ordering.len(),
            numeric_variables.len()
        );
    }

    fn dfs(
        &self,
        from: EitherVar,
        vars: &mut [ExplicitVariable],
        numeric_vars: &mut [NumericVariable],
    ) {
        if let Some(preds) = self.predecessor_graph.get(&from) {
            for pred in preds.keys() {
                let curr_predecessor = *pred;
                match curr_predecessor {
                    EitherVar::ExplicitVariable(v) => {
                        if !vars[v].is_necessary() {
                            vars[v].set_necessary();
                            if DEBUG {
                                println!("var {} is necessary.", vars[v].get_name());
                            }
                            self.dfs(curr_predecessor, vars, numeric_vars);
                        }
                    }
                    EitherVar::NumericVariable(v) => {
                        if !numeric_vars[v].is_necessary() {
                            numeric_vars[v].set_necessary();
                            if DEBUG {
                                println!("var {} is necessary.", numeric_vars[v].get_name());
                            }
                            self.dfs(curr_predecessor, vars, numeric_vars);
                        }
                    }
                }
            }
        }
    }

    fn set_variable_instrumentation_necessary(
        &self,
        numeric_variables: &mut [NumericVariable],
        inst_var: usize,
    ) {
        if !numeric_variables[inst_var].is_necessary() {
            if DEBUG {
                println!(
                    "{} is necessary for the metric",
                    numeric_variables[inst_var].get_name()
                );
            }
            numeric_variables[inst_var].set_instrumentation();
        }
        for ass_ax in self.ass_axioms.iter() {
            if inst_var == ass_ax.get_effect_var() {
                self.set_variable_instrumentation_necessary(
                    numeric_variables,
                    ass_ax.get_left_var(),
                );
                self.set_variable_instrumentation_necessary(
                    numeric_variables,
                    ass_ax.get_right_var(),
                );
            }
        }
    }

    pub fn check_and_repair_empty_axiom_layers(&self) {
        let mut max_num_index_before = -1;
        let mut max_num_index_after = -1;
        for nvar in self.numeric_variables.borrow().iter() {
            max_num_index_before = max(max_num_index_before, nvar.get_layer());
            if nvar.is_necessary() {
                max_num_index_after = max(max_num_index_after, nvar.get_layer());
            }
        }
        if max_num_index_before != max_num_index_after {
            if DEBUG {
                println!(
                    "index before = {} after = {}",
                    max_num_index_before, max_num_index_after
                );
            }
            let decrement = max_num_index_before - max_num_index_after;

            let mut vars = self.variables.borrow_mut();
            for var in vars.iter_mut() {
                var.decrement_layer(decrement);
                assert!(var.get_layer() == -1 || var.get_layer() > max_num_index_after);
            }
        }
    }

    pub fn get_metric_index(&self) -> usize {
        let index = self.numeric_variables.borrow()[self.metric_var].get_level();
        assert!(index >= 0);

        index as usize
    }

    pub fn get_variable_ordering(&self) -> &ExplicitOrderingVars {
        &self.propositional_ordering
    }

    pub fn get_numeric_variable_ordering(&self) -> &NumericOrderingVars {
        &self.numeric_ordering
    }

    pub fn is_acyclic(&self) -> bool {
        self.acyclic
    }

    pub fn dump(&self) {
        for (source, succs) in &self.weighted_graph {
            println!(
                "dependent on var {}: ",
                source.get_name(&self.variables.borrow(), &self.numeric_variables.borrow())
            );
            for (succ, weight) in succs {
                println!(
                    "  [{}, {}]",
                    succ.get_name(&self.variables.borrow(), &self.numeric_variables.borrow()),
                    weight
                );
            }
        }
        for (source, succs) in &self.predecessor_graph {
            println!(
                "var {} is dependent of: ",
                source.get_name(&self.variables.borrow(), &self.numeric_variables.borrow())
            );
            for (succ, weight) in succs {
                println!(
                    "  [{}, {}]",
                    succ.get_name(&self.variables.borrow(), &self.numeric_variables.borrow()),
                    weight
                );
            }
        }
    }

    pub fn strip_operators(&mut self) {
        let old_count = self.operators.len();
        for op in self.operators.iter_mut() {
            op.strip_unimportant_effects(
                &self.variables.borrow(),
                &self.numeric_variables.borrow(),
            );
        }
        self.operators
            .retain(|op| !op.is_redundant(&self.numeric_variables.borrow()));
        println!(
            "{} of {} operators necessary.",
            self.operators.len(),
            old_count
        );
    }

    pub fn strip_mutexes(&mut self) {
        let old_count = self.mutexes.len();
        for mutex in self.mutexes.iter_mut() {
            mutex.strip_unimportant_facts(&self.variables.borrow());
        }
        self.mutexes.retain(|m| !m.is_redundant());
        println!(
            "{} of {} mutex groups necessary.",
            self.mutexes.len(),
            old_count
        );
    }

    pub fn strip_axiom_relationals(&mut self) {
        let old_count = self.axioms.len();
        self.axioms
            .retain(|axiom| !axiom.is_redundant(&self.variables.borrow()));
        println!(
            "{} of {} axiom rules necessary.",
            self.axioms.len(),
            old_count
        );
    }

    pub fn strip_axiom_functional_comparisons(&mut self) {
        let old_count = self.comp_axioms.len();
        self.comp_axioms.retain(|axiom| {
            !axiom.is_redundant(&self.variables.borrow(), &self.numeric_variables.borrow())
        });
        println!(
            "{} of {} axiom_functional assignment rules necessary.",
            self.comp_axioms.len(),
            old_count
        );
    }

    pub fn strip_axiom_functional_assignment(&mut self) {
        let old_count = self.ass_axioms.len();
        self.ass_axioms
            .retain(|axiom| !axiom.is_redundant(&self.numeric_variables.borrow()));
        println!(
            "{} of {} axiom_functional comparison rules necessary.",
            self.ass_axioms.len(),
            old_count
        );
    }

    #[allow(clippy::type_complexity)]
    pub fn finalize(
        mut self,
    ) -> (
        Vec<ExplicitVariable>,
        Vec<NumericVariable>,
        Vec<ExplicitVariable>,
        Vec<NumericVariable>,
        Vec<Operator>,
        Vec<AxiomRelational>,
        Vec<AxiomNumericComputation>,
        Vec<AxiomFunctionalComparison>,
        Vec<MutexGroup>,
        Vec<ExplicitFact>,
        Option<GlobalConstraint>,
        bool,
        usize,
    ) {
        self.strip_mutexes();
        self.strip_operators();
        self.strip_axiom_relationals();
        self.strip_axiom_functional_comparisons();
        self.strip_axiom_functional_assignment();

        self.check_and_repair_empty_axiom_layers();

        let is_acyclic = self.is_acyclic();
        let metric_index = self.get_metric_index();

        let variables: Vec<ExplicitVariable> = self.variables.take();
        let numeric_variables: Vec<NumericVariable> = self.numeric_variables.take();
        let mut ordered_vars = Vec::with_capacity(self.propositional_ordering.len());
        for v in &self.propositional_ordering {
            ordered_vars.push(variables[*v].clone());
        }
        let mut ordered_numeric_vars = Vec::with_capacity(self.numeric_ordering.len());
        for v in &self.numeric_ordering {
            ordered_numeric_vars.push(numeric_variables[*v].clone());
        }

        (
            variables,
            numeric_variables,
            ordered_vars,
            ordered_numeric_vars,
            self.operators,
            self.axioms,
            self.ass_axioms,
            self.comp_axioms,
            self.mutexes,
            self.goals,
            self.global_constraint,
            is_acyclic,
            metric_index,
        )
    }
}
