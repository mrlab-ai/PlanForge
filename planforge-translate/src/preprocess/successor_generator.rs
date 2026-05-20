use std::io::Write;

use tracing::debug;

use super::Condition;
use super::fact::ExplicitFact;
use super::operator::{Operator, PrePost, Prevail};
use super::variable::ExplicitVariable;

#[derive(Debug)]
enum GeneratorBase {
    Switch(GeneratorSwitch),
    Leaf(GeneratorLeaf),
    Empty(GeneratorEmpty),
}

impl GeneratorBase {
    fn dump(&self, indent: &str, vars: &[ExplicitVariable]) {
        match self {
            GeneratorBase::Switch(s) => s.dump(indent, vars),
            GeneratorBase::Leaf(l) => l.dump(indent),
            GeneratorBase::Empty(e) => e.dump(indent),
        }
    }

    fn to_sas<W: Write>(&self, out: &mut W, vars: &[ExplicitVariable]) {
        match self {
            GeneratorBase::Switch(s) => s.to_sas(out, vars),
            GeneratorBase::Leaf(l) => l.to_sas(out),
            GeneratorBase::Empty(e) => e.generate_cpp_input(out),
        }
    }
}

#[derive(Debug)]
struct GeneratorSwitch {
    switch_var: usize,
    immediate_ops_indices: Vec<usize>,
    generator_for_value: Vec<GeneratorBase>,
    default_generator: Box<GeneratorBase>,
}

impl GeneratorSwitch {
    fn new(
        switch_var: usize,
        operators: &mut Vec<usize>,
        gen_for_val: Vec<GeneratorBase>,
        default_gen: GeneratorBase,
    ) -> Self {
        let mut immediate_ops_indices = Vec::new();
        immediate_ops_indices.append(operators);
        Self {
            switch_var,
            immediate_ops_indices,
            generator_for_value: gen_for_val,
            default_generator: Box::new(default_gen),
        }
    }

    fn dump(&self, indent: &str, vars: &[ExplicitVariable]) {
        let var = self.switch_var;
        debug!("{}switch on {}", indent, vars[var].get_name());
        debug!("{}immediately:", indent);
        for op_id in &self.immediate_ops_indices {
            debug!("{}{}", indent, op_id);
        }
        for i in 0..vars[var].get_range() {
            debug!("{}case {}:", indent, i);
            self.generator_for_value[i].dump(&format!("{}  ", indent), vars);
        }
        debug!("{}always:", indent);
        self.default_generator.dump(&format!("{}  ", indent), vars);
    }

    fn to_sas<W: Write>(&self, out: &mut W, vars: &[ExplicitVariable]) {
        let var = self.switch_var;
        let level = vars[var].get_level();
        assert!(level != -1);
        writeln!(out, "switch {}", level).unwrap();
        writeln!(out, "check {}", self.immediate_ops_indices.len()).unwrap();
        for op_id in &self.immediate_ops_indices {
            writeln!(out, "{}", op_id).unwrap();
        }
        for i in 0..vars[var].get_range() {
            self.generator_for_value[i].to_sas(out, vars);
        }
        self.default_generator.to_sas(out, vars);
    }
}

#[derive(Debug)]
struct GeneratorLeaf {
    applicable_ops_indices: Vec<usize>,
}

impl GeneratorLeaf {
    fn new(operators: &mut Vec<usize>) -> Self {
        let mut applicable_ops_indices = Vec::new();
        applicable_ops_indices.append(operators);
        Self {
            applicable_ops_indices,
        }
    }

    fn dump(&self, indent: &str) {
        for op_id in &self.applicable_ops_indices {
            debug!("{}{}", indent, op_id);
        }
    }

    fn to_sas<W: Write>(&self, out: &mut W) {
        writeln!(out, "check {}", self.applicable_ops_indices.len()).unwrap();
        for op_id in &self.applicable_ops_indices {
            writeln!(out, "{}", op_id).unwrap();
        }
    }
}

#[derive(Debug)]
struct GeneratorEmpty;

impl GeneratorEmpty {
    fn dump(&self, indent: &str) {
        debug!("{}<empty>", indent);
    }

    fn generate_cpp_input<W: Write>(&self, out: &mut W) {
        writeln!(out, "check 0").unwrap();
    }
}

#[derive(Debug, Default)]
pub struct SuccessorGenerator {
    root: Option<GeneratorBase>,
    conditions: Vec<Condition>,
    next_condition_by_op: Vec<usize>,
    var_order: Vec<ExplicitVariable>,
}

impl SuccessorGenerator {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_vars_and_ops(mut variables: Vec<ExplicitVariable>, operators: &[Operator]) -> Self {
        let num_operators = operators.len();
        let mut conditions: Vec<Condition> = Vec::with_capacity(num_operators);
        let mut next_condition_by_op: Vec<usize> = Vec::with_capacity(num_operators);
        let mut all_operator_indices: Vec<usize> = Vec::new();

        for (i, op) in operators.iter().enumerate() {
            let mut cond: Condition = Vec::new();
            for Prevail { var, prev } in op.get_prevail() {
                cond.push(ExplicitFact {
                    var: *var,
                    value: *prev,
                });
            }
            for PrePost { var, pre, .. } in op.get_pre_post() {
                if pre.is_some() {
                    cond.push(ExplicitFact {
                        var: *var,
                        value: pre.unwrap(),
                    });
                }
            }
            cond.sort();
            all_operator_indices.push(i);
            conditions.push(cond);
            next_condition_by_op.push(0);
        }

        variables.sort();

        let mut sg = SuccessorGenerator {
            root: None,
            conditions,
            next_condition_by_op,
            var_order: variables,
        };

        let root = sg.construct_recursive(0, all_operator_indices);
        sg.root = Some(root);
        sg
    }

    #[allow(clippy::needless_range_loop)]
    fn construct_recursive(
        &mut self,
        mut switch_var_no: usize,
        mut op_indices: Vec<usize>,
    ) -> GeneratorBase {
        if op_indices.is_empty() {
            return GeneratorBase::Empty(GeneratorEmpty);
        }
        let num_vars = self.var_order.len();

        loop {
            if switch_var_no == num_vars {
                return GeneratorBase::Leaf(GeneratorLeaf::new(&mut op_indices));
            }

            let switch_var = switch_var_no;
            let number_of_children = self.var_order[switch_var].get_range();

            let mut ops_for_val_indices: Vec<Vec<usize>> = vec![Vec::new(); number_of_children];
            let mut default_ops_indices: Vec<usize> = Vec::new();
            let mut applicable_ops_indices: Vec<usize> = Vec::new();

            let mut all_ops_are_immediate = true;
            let mut var_is_interesting = false;

            while let Some(op_index) = op_indices.pop() {
                let op_idx_usize = op_index;
                let cond_iter = self.next_condition_by_op[op_idx_usize];
                let conds = &self.conditions[op_idx_usize];
                if cond_iter == conds.len() {
                    var_is_interesting = true;
                    applicable_ops_indices.push(op_index);
                } else {
                    all_ops_are_immediate = false;
                    let fact = &conds[cond_iter];
                    if fact.var == switch_var {
                        var_is_interesting = true;
                        let mut next_idx = cond_iter;
                        while next_idx < conds.len() && conds[next_idx].var == switch_var {
                            next_idx += 1;
                        }
                        self.next_condition_by_op[op_idx_usize] = next_idx;
                        ops_for_val_indices[fact.value].push(op_index);
                    } else {
                        default_ops_indices.push(op_index);
                    }
                }
            }

            if all_ops_are_immediate {
                return GeneratorBase::Leaf(GeneratorLeaf::new(&mut applicable_ops_indices));
            } else if var_is_interesting {
                let mut gen_for_val: Vec<GeneratorBase> = Vec::new();
                for j in 0..number_of_children {
                    let child =
                        self.construct_recursive(switch_var_no + 1, ops_for_val_indices[j].clone());
                    gen_for_val.push(child);
                }
                let default_sg = self.construct_recursive(switch_var_no + 1, default_ops_indices);
                return GeneratorBase::Switch(GeneratorSwitch::new(
                    switch_var,
                    &mut applicable_ops_indices,
                    gen_for_val,
                    default_sg,
                ));
            } else {
                switch_var_no += 1;
                op_indices = default_ops_indices;
            }
        }
    }

    pub fn dump(&self, vars: &[ExplicitVariable]) {
        debug!("Successor Generator:");
        if let Some(root) = &self.root {
            root.dump("  ", vars);
        }
    }

    pub fn to_sas<W: Write>(&self, out: &mut W, vars: &[ExplicitVariable]) {
        if let Some(root) = &self.root {
            root.to_sas(out, vars);
        }
    }
}
