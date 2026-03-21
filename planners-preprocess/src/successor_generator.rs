use std::io::Write;

use crate::operator::{Operator, PrePost, Prevail};
use crate::variable::Variable;

type Condition = Vec<(*const Variable, i32)>;

#[derive(Debug)]
enum GeneratorBase {
    Switch(GeneratorSwitch),
    Leaf(GeneratorLeaf),
    Empty(GeneratorEmpty),
}

impl GeneratorBase {
    fn dump(&self, indent: &str) {
        match self {
            GeneratorBase::Switch(s) => s.dump(indent),
            GeneratorBase::Leaf(l) => l.dump(indent),
            GeneratorBase::Empty(e) => e.dump(indent),
        }
    }

    fn generate_cpp_input<W: Write>(&self, out: &mut W) {
        match self {
            GeneratorBase::Switch(s) => s.generate_cpp_input(out),
            GeneratorBase::Leaf(l) => l.generate_cpp_input(out),
            GeneratorBase::Empty(e) => e.generate_cpp_input(out),
        }
    }
}

#[derive(Debug)]
struct GeneratorSwitch {
    switch_var: *const Variable,
    immediate_ops_indices: Vec<i32>,
    generator_for_value: Vec<GeneratorBase>,
    default_generator: Box<GeneratorBase>,
}

impl GeneratorSwitch {
    fn new(
        switch_var: *const Variable,
        operators: &mut Vec<i32>,
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

    fn dump(&self, indent: &str) {
        let var = unsafe { &*self.switch_var };
        println!("{}switch on {}", indent, var.get_name());
        println!("{}immediately:", indent);
        for op_id in &self.immediate_ops_indices {
            println!("{}{}", indent, op_id);
        }
        for i in 0..var.get_range() {
            println!("{}case {}:", indent, i);
            self.generator_for_value[i as usize].dump(&format!("{}  ", indent));
        }
        println!("{}always:", indent);
        self.default_generator.dump(&format!("{}  ", indent));
    }

    fn generate_cpp_input<W: Write>(&self, out: &mut W) {
        let var = unsafe { &*self.switch_var };
        let level = var.get_level();
        assert!(level != -1);
        writeln!(out, "switch {}", level).unwrap();
        writeln!(out, "check {}", self.immediate_ops_indices.len()).unwrap();
        for op_id in &self.immediate_ops_indices {
            writeln!(out, "{}", op_id).unwrap();
        }
        for i in 0..var.get_range() {
            self.generator_for_value[i as usize].generate_cpp_input(out);
        }
        self.default_generator.generate_cpp_input(out);
    }
}

#[derive(Debug)]
struct GeneratorLeaf {
    applicable_ops_indices: Vec<i32>,
}

impl GeneratorLeaf {
    fn new(operators: &mut Vec<i32>) -> Self {
        let mut applicable_ops_indices = Vec::new();
        applicable_ops_indices.append(operators);
        Self {
            applicable_ops_indices,
        }
    }

    fn dump(&self, indent: &str) {
        for op_id in &self.applicable_ops_indices {
            println!("{}{}", indent, op_id);
        }
    }

    fn generate_cpp_input<W: Write>(&self, out: &mut W) {
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
        println!("{}<empty>", indent);
    }

    fn generate_cpp_input<W: Write>(&self, out: &mut W) {
        writeln!(out, "check 0").unwrap();
    }
}

#[derive(Debug)]
pub struct SuccessorGenerator {
    root: Option<GeneratorBase>,
    conditions: Vec<Condition>,
    next_condition_by_op: Vec<usize>,
    var_order: Vec<*const Variable>,
}

impl SuccessorGenerator {
    pub fn new() -> Self {
        Self {
            root: None,
            conditions: Vec::new(),
            next_condition_by_op: Vec::new(),
            var_order: Vec::new(),
        }
    }

    pub fn from_vars_and_ops(variables: &[&Variable], operators: &[Operator]) -> Self {
        let num_operators = operators.len();
        let mut conditions: Vec<Condition> = Vec::with_capacity(num_operators);
        let mut next_condition_by_op: Vec<usize> = Vec::with_capacity(num_operators);
        let mut all_operator_indices: Vec<i32> = Vec::new();

        for (i, op) in operators.iter().enumerate() {
            let mut cond: Condition = Vec::new();
            for Prevail { var, prev } in op.get_prevail() {
                cond.push((*var, *prev));
            }
            for PrePost { var, pre, .. } in op.get_pre_post() {
                if *pre != -1 {
                    cond.push((*var, *pre));
                }
            }
            cond.sort_by(|a, b| {
                let ap = a.0 as usize;
                let bp = b.0 as usize;
                if ap == bp {
                    a.1.cmp(&b.1)
                } else {
                    ap.cmp(&bp)
                }
            });
            all_operator_indices.push(i as i32);
            conditions.push(cond);
            next_condition_by_op.push(0);
        }

        let mut var_order: Vec<*const Variable> =
            variables.iter().map(|v| *v as *const Variable).collect();
        var_order.sort_by(|a, b| (*a as usize).cmp(&(*b as usize)));

        let mut sg = SuccessorGenerator {
            root: None,
            conditions,
            next_condition_by_op,
            var_order,
        };

        let root = sg.construct_recursive(0, all_operator_indices);
        sg.root = Some(root);
        sg
    }

    fn construct_recursive(
        &mut self,
        mut switch_var_no: usize,
        mut op_indices: Vec<i32>,
    ) -> GeneratorBase {
        if op_indices.is_empty() {
            return GeneratorBase::Empty(GeneratorEmpty);
        }
        let num_vars = self.var_order.len();

        loop {
            if switch_var_no == num_vars {
                return GeneratorBase::Leaf(GeneratorLeaf::new(&mut op_indices));
            }

            let switch_var = self.var_order[switch_var_no];
            let number_of_children = unsafe { &*switch_var }.get_range();

            let mut ops_for_val_indices: Vec<Vec<i32>> =
                vec![Vec::new(); number_of_children as usize];
            let mut default_ops_indices: Vec<i32> = Vec::new();
            let mut applicable_ops_indices: Vec<i32> = Vec::new();

            let mut all_ops_are_immediate = true;
            let mut var_is_interesting = false;

            while let Some(op_index) = op_indices.pop() {
                let op_idx_usize = op_index as usize;
                let cond_iter = self.next_condition_by_op[op_idx_usize];
                let conds = &self.conditions[op_idx_usize];
                if cond_iter == conds.len() {
                    var_is_interesting = true;
                    applicable_ops_indices.push(op_index);
                } else {
                    all_ops_are_immediate = false;
                    let (var, val) = conds[cond_iter];
                    if var == switch_var {
                        var_is_interesting = true;
                        let mut next_idx = cond_iter;
                        while next_idx < conds.len() && conds[next_idx].0 == switch_var {
                            next_idx += 1;
                        }
                        self.next_condition_by_op[op_idx_usize] = next_idx;
                        ops_for_val_indices[val as usize].push(op_index);
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
                    let child = self.construct_recursive(
                        switch_var_no + 1,
                        ops_for_val_indices[j as usize].clone(),
                    );
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

    pub fn dump(&self) {
        println!("Successor Generator:");
        if let Some(root) = &self.root {
            root.dump("  ");
        }
    }

    pub fn generate_cpp_input<W: Write>(&self, out: &mut W) {
        if let Some(root) = &self.root {
            root.generate_cpp_input(out);
        }
    }
}
